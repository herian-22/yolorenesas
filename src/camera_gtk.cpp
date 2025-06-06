void camera_thread();
#include <opencv2/opencv.hpp>
#include <gtk/gtk.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/dnn.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <thread>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <pthread.h>

// Forward declarations
static gboolean update_image(gpointer data);
static gboolean update_stats(gpointer data);
static void on_window_destroy(GtkWidget* widget, gpointer data);
static gboolean create_window(gpointer data);

// Struktur untuk statistik performa
struct PerformanceStats {
    double fps = 0.0;
    double inference_time = 0.0;
    double boot_time = 0.0;
    int cpu_cores = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_frame_time;
    int frame_count = 0;
};

static GtkWidget* window = nullptr;
static GtkWidget* image_widget = nullptr;
static GtkWidget* stats_label = nullptr;
static bool running = true;
cv::Mat frame_rgb;
std::mutex frame_mutex;
cv::VideoCapture cap;
PerformanceStats stats;
cv::dnn::Net net;
Ort::Env* ort_env = nullptr;
Ort::Session* ort_session = nullptr;
Ort::SessionOptions ort_session_options;
std::vector<std::string> ort_input_names;
std::vector<std::string> ort_output_names;

// Fungsi untuk mendapatkan jumlah core CPU
int get_cpu_cores() {
    return std::thread::hardware_concurrency();
}

// Fungsi untuk inisialisasi YOLOv5 dengan ONNXRuntime
bool init_yolo() {
    try {
        ort_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "yolo");
        ort_session_options.SetIntraOpNumThreads(stats.cpu_cores);
        ort_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        ort_session = new Ort::Session(*ort_env, "yolov5s_fp32.onnx", ort_session_options);
        
        // Ambil nama input/output menggunakan API yang benar
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Debug: Print jumlah input dan output
        size_t num_input_nodes = ort_session->GetInputCount();
        size_t num_output_nodes = ort_session->GetOutputCount();
        g_print("Number of inputs: %zu\n", num_input_nodes);
        g_print("Number of outputs: %zu\n", num_output_nodes);
        
        // Ambil nama input/output
        auto input_name = ort_session->GetInputNameAllocated(0, allocator);
        auto output_name = ort_session->GetOutputNameAllocated(0, allocator);
        
        // Debug: Print nama input/output
        g_print("Input name: %s\n", input_name.get());
        g_print("Output name: %s\n", output_name.get());
        
        // Simpan nama input/output sebagai string
        ort_input_names.clear();
        ort_output_names.clear();
        ort_input_names.push_back(std::string(input_name.get()));
        ort_output_names.push_back(std::string(output_name.get()));
        
        return true;
    } catch (const Ort::Exception& e) {
        g_print("Error loading YOLO model (ONNXRuntime): %s\n", e.what());
        return false;
    }
}

// Tambahkan fungsi post-processing YOLOv5
void post_process_yolo(float* output, int rows, int cols, float conf_threshold, float iou_threshold, std::vector<cv::Rect>& boxes, std::vector<float>& scores, std::vector<int>& class_ids) {
    // Output shape: [1, 25200, 85] (YOLOv5s)
    // 85 = 4 (box) + 1 (conf) + 80 (classes)
    for (int i = 0; i < rows; ++i) {
        float* row = output + i * cols;
        float conf = row[4];
        if (conf < conf_threshold) continue;
        // Ambil class dengan score tertinggi
        int class_id = 0;
        float max_class_score = 0;
        for (int j = 5; j < cols; ++j) {
            if (row[j] > max_class_score) {
                max_class_score = row[j];
                class_id = j - 5;
            }
        }
        if (max_class_score < conf_threshold) continue;
        // Decode box (x, y, w, h) -> (x1, y1, x2, y2)
        float x = row[0];
        float y = row[1];
        float w = row[2];
        float h = row[3];
        int x1 = static_cast<int>((x - w / 2) * 640);
        int y1 = static_cast<int>((y - h / 2) * 640);
        int x2 = static_cast<int>((x + w / 2) * 640);
        int y2 = static_cast<int>((y + h / 2) * 640);
        boxes.push_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
        scores.push_back(conf * max_class_score);
        class_ids.push_back(class_id);
    }
    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, indices);
    std::vector<cv::Rect> nms_boxes;
    std::vector<float> nms_scores;
    std::vector<int> nms_class_ids;
    for (int idx : indices) {
        nms_boxes.push_back(boxes[idx]);
        nms_scores.push_back(scores[idx]);
        nms_class_ids.push_back(class_ids[idx]);
    }
    boxes = nms_boxes;
    scores = nms_scores;
    class_ids = nms_class_ids;
}

// Update main function
int main(int argc, char* argv[]) {
    gtk_init(&argc, &argv);
    
    // Inisialisasi YOLO
    if (!init_yolo()) {
        g_print("Failed to initialize YOLO\n");
        return -1;
    }

    // Inisialisasi stats
    stats.frame_count = 0;
    stats.fps = 0;
    stats.inference_time = 0;
    stats.last_frame_time = std::chrono::high_resolution_clock::now();
    stats.cpu_cores = get_cpu_cores();

    // Buat window di thread utama
    create_window(NULL);

    // Mulai thread kamera
    std::thread cam_thread(camera_thread);

    // Jalankan main loop GTK
    gtk_main();

    // Tunggu thread selesai
    cam_thread.join();

    return 0;
}

// Update camera_thread
void camera_thread() {
    g_print("Opening camera...\n");
    cv::VideoCapture cap;
    g_print("Trying to open camera with V4L2...\n");
    cap.open(0, cv::CAP_V4L2);

    if (!cap.isOpened()) {
        g_print("Failed to open camera with V4L2, trying default backend...\n");
        cap.open(0);
    }

    if (!cap.isOpened()) {
        g_print("Error: Could not open camera with any backend\n");
        return;
    }

    g_print("Camera opened successfully\n");

    // Set properti kamera
    g_print("Setting camera properties...\n");
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    // Verifikasi properti kamera
    g_print("Camera properties:\n");
    g_print("Width: %d\n", (int)cap.get(cv::CAP_PROP_FRAME_WIDTH));
    g_print("Height: %d\n", (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    g_print("FPS: %d\n", (int)cap.get(cv::CAP_PROP_FPS));
    g_print("Format: %d\n", (int)cap.get(cv::CAP_PROP_FOURCC));

    g_print("Starting camera loop...\n");
    while (running) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            g_print("Error: Empty frame\n");
            break;
        }

        // Preprocessing untuk YOLO
        cv::Mat blob;
        cv::resize(frame, frame, cv::Size(640, 640));
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(640, 640), cv::Scalar(0,0,0), true, false);

        // Siapkan input tensor
        std::vector<float> input_tensor_values(blob.begin<float>(), blob.end<float>());
        std::vector<int64_t> input_shape = {1, 3, 640, 640};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size());

        // Lakukan inferensi
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<const char*> input_names = {ort_input_names[0].c_str()};
        std::vector<const char*> output_names = {ort_output_names[0].c_str()};
        auto output_tensors = ort_session->Run(
            Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
            output_names.data(), output_names.size());
        auto end_time = std::chrono::high_resolution_clock::now();
        stats.inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        // Post-processing hasil deteksi
        float* output = output_tensors[0].GetTensorMutableData<float>();
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;
        post_process_yolo(output, 25200, 85, 0.25, 0.45, boxes, scores, class_ids);

        // Gambar bounding box
        for (size_t i = 0; i < boxes.size(); i++) {
            cv::rectangle(frame, boxes[i], cv::Scalar(0, 255, 0), 2);
            std::string label = cv::format("%.2f", scores[i]);
            cv::putText(frame, label, cv::Point(boxes[i].x, boxes[i].y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        // Convert frame to GTK format and update display
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        GdkPixbuf* pixbuf = gdk_pixbuf_new_from_data(
            frame.data, GDK_COLORSPACE_RGB, false, 8,
            frame.cols, frame.rows, frame.step, NULL, NULL);
        
        if (pixbuf) {
            gdk_threads_add_idle(update_image, pixbuf);
        } else {
            g_print("Error creating pixbuf\n");
        }

        // Update stats
        char stats_text[256];
        snprintf(stats_text, sizeof(stats_text), 
                 "FPS: %.1f | Inference: %.1f ms | CPU Cores: %d",
                 stats.fps, stats.inference_time, stats.cpu_cores);
        gdk_threads_add_idle(update_stats, g_strdup(stats_text));

        // Update FPS counter
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - stats.last_frame_time).count();
        if (elapsed >= 1000) {
            stats.fps = stats.frame_count * 1000.0 / elapsed;
            stats.frame_count = 0;
            stats.last_frame_time = now;
        }
        stats.frame_count++;

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    cap.release();
}

// Fungsi untuk menangani window destroy
static void on_window_destroy(GtkWidget* widget, gpointer data) {
    running = false;
    gtk_main_quit();
}

// Fungsi untuk membuat window
static gboolean create_window(gpointer data) {
    g_print("Creating GTK window...\n");
    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "Camera View");
    gtk_window_set_default_size(GTK_WINDOW(window), 800, 600);
    g_signal_connect(window, "destroy", G_CALLBACK(on_window_destroy), NULL);

    // Buat box untuk layout
    GtkWidget* box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_container_add(GTK_CONTAINER(window), box);

    // Buat image widget
    image_widget = gtk_image_new();
    gtk_box_pack_start(GTK_BOX(box), image_widget, TRUE, TRUE, 0);

    // Buat label untuk stats
    stats_label = gtk_label_new("");
    gtk_box_pack_start(GTK_BOX(box), stats_label, FALSE, FALSE, 0);

    // Tampilkan window
    gtk_widget_show_all(window);
    g_print("GTK window created and shown\n");

    return G_SOURCE_REMOVE;
}

// Update fungsi update_image
static gboolean update_image(gpointer data) {
    GdkPixbuf* pixbuf = (GdkPixbuf*)data;
    if (pixbuf && image_widget) {
        gtk_image_set_from_pixbuf(GTK_IMAGE(image_widget), pixbuf);
        g_object_unref(pixbuf);
    }
    return G_SOURCE_REMOVE;
}

// Update fungsi update_stats
static gboolean update_stats(gpointer data) {
    char* text = (char*)data;
    if (text && stats_label) {
        gtk_label_set_text(GTK_LABEL(stats_label), text);
        g_free(text);
    }
    return G_SOURCE_REMOVE;
}
