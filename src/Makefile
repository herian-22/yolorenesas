CXX = g++
CXXFLAGS = -O3 -pthread
INCLUDES = -I/usr/local/onnxruntime/include \
           $(shell pkg-config --cflags gtk+-3.0) \
           $(shell pkg-config --cflags opencv4)
LDFLAGS = -L/usr/local/onnxruntime/lib \
          $(shell pkg-config --libs gtk+-3.0) \
          $(shell pkg-config --libs opencv4) \
          -lgtk-3 -lglib-2.0 -lgobject-2.0 \
          $(shell pkg-config --libs gdk-pixbuf-2.0) \
          -lonnxruntime

TARGET = camera_gtk
SRC = /media/herboy/OS/Renesas/belajar/src/camera_gtk.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean 