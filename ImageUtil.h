#ifndef IMAGEUTIL_H
#define IMAGEUTIL_H
#include <vector>
#include <string>

//MISC UTILS
int floatToRGB(double r, double g, double b, double a = 0);
//writes all the binary data given to a file.
void writeCharFile(std::string name, std::vector<char> data);

/*Example usage of Image:

  Image x(1280,800)
  x.put(100,100,floatToRGB(1,0,0));
  x.save("myBitmap.bmp");

  Saves a bitmap with a single red dot on it. For some weird reason
  the vertical axis is flipped, but that's not important right now. */
class Image {
    int width;
    int height;

    std::vector<char> data;
    public:
    Image(int w, int h);
    Image(std::vector<double> arr, int w, int h);
    void createFromArray(std::vector<double> arr, int width, int height);
    void put(int x, int y, int c);
    int get(int x, int y);
    void save(std::string filename);
};

/*
   It's convenient sometimes to just keep a 2D array of doubles
   while treating it as an "image".
 */
class DoubleImage {
    int width;
    int height;

    std::vector<double> data;
    public:
    int getWidth();
    int getHeight();
    DoubleImage();
    DoubleImage(int w, int h);
    DoubleImage(std::vector<double> arr, int w, int h);
    void createFromArray(std::vector<double> arr, int width, int height);
    const std::vector<double> getData();
    void put(int x, int y, double c);
    void increase(int x, int y, double c);
    double get(int x, int y);
};
#endif
