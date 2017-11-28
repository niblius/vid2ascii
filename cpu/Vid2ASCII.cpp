#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <string>
#include <cmath>
#include "characters.h"

#define SKIP 0

using namespace cv;
using namespace std;

void get_character_set(Mat &charset);
int32_t calculate_intensity(Mat img);
int fit_by_intensity(int32_t intens[], int32_t intensity);
void calc_font_intensities(int32_t intensities[], Mat &img);
int32_t _calculate_intensity(Mat img);

int main(int argc, char* argv[])
{
    // calculate charset intensities
    Mat font_img;
    get_character_set(font_img);
    int32_t intensities[CHARACTER_COUNT];
    calc_font_intensities(intensities, font_img);

    // get the frame
    string path = "sample.avi";
    VideoCapture capture(path);
    Mat frame, gray_frame;

    int frm_count = capture.get(CV_CAP_PROP_FRAME_COUNT);
    
    char *ascii_img;
    int symbols_per_width;
    int symbols_per_hight;
    int result_size;
    int frame_size;

    // don't calc last frame
    for (int frm_numb = 0; frm_numb < frm_count-1; frm_numb++) {
        // skipping some number of frames
        for (int skiped = 0; skiped < SKIP; skiped++)
            capture.grab();
        
        // load frame
        capture >> frame;
        // convert to gray-scale
        cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
        
        // display current frame
        // imshow("Vid2ASCII", gray_frame);
        // waitKey(1);

        if (frm_numb == 0) {
            symbols_per_width = gray_frame.cols / SYMBOL_WIDTH;
            symbols_per_hight = gray_frame.rows / LINE_HEIGHT;
            
            result_size = symbols_per_hight*symbols_per_width*frm_count;
            frame_size = symbols_per_hight*symbols_per_width;
            ascii_img = new char[result_size];
        }
        
        for (int y = 0; y < symbols_per_hight; y++) {
            for (int x = 0; x < symbols_per_width; x++) {
                Rect char_borders(x*SYMBOL_WIDTH, y*LINE_HEIGHT, SYMBOL_WIDTH, LINE_HEIGHT);
                Mat roi = gray_frame(char_borders);
                int32_t inten = _calculate_intensity(roi);

                int best_fit = fit_by_intensity(intensities, inten);

                /*
                Character c = characters_Arial[best_fit];
                if (symbol_width+x <= gray_frame.cols && line_height+y <= gray_frame.rows) {
                    Rect borders(x, y, symbol_width, line_height);
                    Mat roi = gray_frame(borders);
                    roi.setTo(Scalar(0xff, 0xff, 0xff));
                    imshow("Vid2ASCII", gray_frame);
                }*/

                ascii_img[frm_numb*frame_size + y*symbols_per_width + x] = (char)characters_Arial[best_fit].codePoint;
            }
        }
    }

    FILE *output = fopen("ascii_img.txt", "w");

    for (int f = 0; f < frm_count; f++) {
        for (int y = 0; y < symbols_per_hight; y++) {
            for (int x = 0; x < symbols_per_width; x++) {
                fputc(ascii_img[frame_size*f + symbols_per_width*y + x], output);
            }
            fputc('\n', output);        
        }
        fputc('\n', output);
    }

    fclose(output);
    
    delete[] ascii_img;

    return 0;
}

void calc_font_intensities(int32_t intensities[], Mat &charset) {
    for (int i = 0; i < CHARACTER_COUNT; i++) {
        Character c = characters_Arial[i];
        Rect char_borders(c.x, c.y, c.width, c.height);
        Mat c_img = charset(char_borders);
        intensities[i] = calculate_intensity(c_img);
    }
}

void get_character_set(Mat &char_set) {
    char_set = imread("font.png", CV_LOAD_IMAGE_GRAYSCALE);
    return;
}

int32_t calculate_intensity(Mat img) {
    int nRows = img.rows;
    int nCols = img.cols;

    if (img.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }

    int i,j;
    uchar* p;
    int32_t intensity = 0;
    for (i = 0; i < nRows; ++i)
    {
        for (j = 0; j < nCols; ++j)
        {
            intensity += 255 - img.at<uchar>(i,j);
        }
    }
    
    return intensity;
}

int fit_by_intensity(int32_t intens[], int32_t intensity) {
    int32_t diff, smallest_diff = abs(intens[0] - intensity);
    int smallest_index = 0;
    for (int i = 1; i < CHARACTER_COUNT; i++) {
        diff = abs(intens[i] - intensity);
        if (diff < smallest_diff) {
            smallest_diff = diff;
            smallest_index = i;
        }
    }

    return smallest_index;
}

int32_t _calculate_intensity(Mat img) {
    img = img.clone();
    uchar *data = img.data;

    int i,j;
    int32_t intensity = 0;
    for (i = 0; i < LINE_HEIGHT; ++i)
    {
        for (j = 0; j < SYMBOL_WIDTH; ++j)
        {
            intensity += 255 - data[i*SYMBOL_WIDTH + j];
        }
    }
    
    return intensity;
}