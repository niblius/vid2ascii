#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <string>
#include <cmath>
#include "characters.h"

using namespace cv;
using namespace std;

void get_character_set(Mat &charset);
int32_t calculate_intensity(Mat img, int light = 0);
int fit_by_intensity(int32_t intens[], int32_t intensity);
void calc_font_intensities(int32_t intensities[], Mat &img);

int main(int argc, char* argv[])
{
    if (argc < 3) {
        cout << "Wrong number of arguments. Usage: Vid2ASCII sample.avi output.txt [VISUALIZATION]"
             << endl;
        return -1;
    }
    
    int VISUALIZATION = 0;
    if (argc > 3) {
        VISUALIZATION = 1;
    }

    // calculate charset intensities
    Mat font_img;
    get_character_set(font_img);
    int32_t intensities[CHARACTER_COUNT];
    calc_font_intensities(intensities, font_img);

    // get the frame
    VideoCapture capture(argv[1]);
    Mat frame, gray_frame;
    
    int frame_width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    int frm_total_count = capture.get(CV_CAP_PROP_FRAME_COUNT);
    int frm_total_processed = 0,
        frm_count = 0;
    int allocation_sz = (frm_total_count > 512) ? 512 : frm_total_count;
    
    // All frames have same size
    size_t  symbols_per_width = frame_width / SYMBOL_WIDTH,
        symbols_per_height = frame_height / LINE_HEIGHT;
    int frame_size = symbols_per_height * symbols_per_width;

    // Single interation result size
    int interation_result_sz = frame_size*allocation_sz;

    char *ascii_img = new char[interation_result_sz];
    
    
    FILE *output = fopen(argv[2], "w");

    while (frm_total_processed < frm_total_count) {
        frm_total_processed += frm_count;
        if (frm_total_count-frm_total_processed > 512) {
            frm_count = 512;
        } else {
            frm_count = frm_total_count-frm_total_processed;
        }

        // don't calc last frame
        for (int frm_numb = 0; frm_numb < frm_count; frm_numb++) {
            // load frame
            capture >> frame;
            // convert to gray-scale
            cvtColor(frame, gray_frame, COLOR_BGR2GRAY);

            // There is bug with some videos,
            // OpenCV thinks that there are more frames, than there are really
            // So that those last "broken" frames have dimentions 0x0
            if (frame.rows == 0 || frame.cols == 0) {
                std::cout << frm_total_processed + frm_numb
                    << " is an empty frame, skipping the rest." << std::endl;
                frm_total_processed = frm_total_count;
                break;
            }

            for (int y = 0; y < symbols_per_height; y++) {
                for (int x = 0; x < symbols_per_width; x++) {
                    int roi_x = x*SYMBOL_WIDTH, roi_y = y*LINE_HEIGHT;
                    Rect roi_borders(roi_x, roi_y, SYMBOL_WIDTH, LINE_HEIGHT);
                    Mat roi = gray_frame(roi_borders);
                    int32_t inten = calculate_intensity(roi);

                    int best_fit = fit_by_intensity(intensities, inten);
                    
                    Character c = characters_Arial[best_fit];

                    if (VISUALIZATION) {                        
                        if (roi_x + c.width < frame_width
                            && roi_y + c.height < frame_height) {
                            roi.setTo(Scalar(0xff));
                            Rect character_borders(c.x, c.y, c.width, c.height);
                            Mat character_roi(gray_frame(Rect(roi_x, roi_y, c.width, c.height)));
                            Mat character_img = font_img(character_borders);
                            character_img.copyTo(character_roi);
                            imshow("Vid2ASCII", gray_frame);
                            waitKey(1);
                        }
                    }

                    ascii_img[frm_numb*frame_size + y*symbols_per_width + x] = c.codePoint;
                }
            }
        }

        for (int f = 0; f < frm_count; f++) {
            for (int y = 0; y < symbols_per_height; y++) {
                for (int x = 0; x < symbols_per_width; x++) {
                    fputc(ascii_img[frame_size*f + symbols_per_width*y + x], output);
                }
                fputc('\n', output);        
            }
            fputc('\n', output);
        }
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

int32_t calculate_intensity(Mat img, int light) {
    int nRows = img.rows;
    int nCols = img.cols;

    int max_intens = 255;
    if (light) {
        max_intens = 285;
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