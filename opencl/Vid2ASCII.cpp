#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <CL/cl.h>
#include "characters.h"


void get_character_set(cv::Mat &charset);
int32_t calculate_intensity(cv::Mat img);
void calc_font_intensities(int32_t intensities[], cv::Mat &img);
std::string getPlatformName (cl_platform_id id);
void checkError (cl_int error);
std::string getDeviceName (cl_device_id id);
std::string loadKernel (const char* name);
cl_program createProgram (const std::string& source, cl_context context);


int main()
{
	cl_uint platformIdCount = 0;
	clGetPlatformIDs(0, nullptr, &platformIdCount);

	if (platformIdCount == 0) {
		std::cerr << "No OpenCL platform found" << std::endl;
		return 1;
	} else {
		std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
	}

	std::vector<cl_platform_id> platformIds (platformIdCount);
	clGetPlatformIDs(platformIdCount, platformIds.data (), nullptr);

    // Print names of all platforms
	for (cl_uint i = 0; i < platformIdCount; ++i) {
		std::cout << "\t (" << (i+1) << ") : " << getPlatformName (platformIds [i]) << std::endl;
	}

	cl_uint deviceIdCount = 0;
	clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_ALL, 0, nullptr,
		&deviceIdCount);

	if (deviceIdCount == 0) {
		std::cerr << "No OpenCL devices found" << std::endl;
		return 1;
	} else {
		std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
	}

    // Print names of all devices
	std::vector<cl_device_id> deviceIds (deviceIdCount);
	clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_ALL, deviceIdCount,
		deviceIds.data (), nullptr);

	for (cl_uint i = 0; i < deviceIdCount; ++i) {
		std::cout << "\t (" << (i+1) << ") : " << getDeviceName (deviceIds [i]) << std::endl;
	}

    // Create context
	const cl_context_properties contextProperties [] =
	{
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds [0]),
		0, 0
	};

    // Pass all devices to context
	cl_int error = CL_SUCCESS;
	cl_context context = clCreateContext (contextProperties, deviceIdCount,
		deviceIds.data (), nullptr, nullptr, &error);
	checkError (error);

	std::cout << "Context created" << std::endl;

    // Get video
    std::string path = "sample.avi";
    cv::VideoCapture capture(path);

    // Capture all but last 30 frames
    int frm_count = capture.get(CV_CAP_PROP_FRAME_COUNT)-30;
    cv::Mat frame, *gray_frames;
    // we will load all frames to memory
    // to be able to pre-process frames on CPU
    // while asyncroniously copying them to GPU __global
    gray_frames = new cv::Mat[frm_count];
    cl_mem *gpu_gray = new cl_mem[frm_count];
    cl_mem *gpu_output = new cl_mem[frm_count];
    int symbols_per_width, symbols_per_height;

    std::cout << "Processing " << frm_count << " frames" << std::endl;
    // ------------------------

    if (frm_count > 1000)
        frm_count = 3000;

    // ------------------------

    // Transfer Mat data to the device
    for (int i = 0; i < frm_count; i++) {
        capture >> frame;
        cv::cvtColor(frame, gray_frames[i], cv::COLOR_BGR2GRAY);

        if (i == 0) {
            // All frames have same size
            symbols_per_width = gray_frames[0].cols / SYMBOL_WIDTH,
            symbols_per_height = gray_frames[0].rows / LINE_HEIGHT;
        }

       	// Create a buffers for each frame
        gpu_gray[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            gray_frames[i].total(), gray_frames[i].data, &error);
        checkError(error);

        gpu_output[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            sizeof(char) * symbols_per_width * symbols_per_height,
            nullptr, &error);
        checkError (error);
    }

    // Calculate charset intensities
    cv::Mat font_img;
    get_character_set(font_img);
    int32_t intensities[CHARACTER_COUNT];
    calc_font_intensities(intensities, font_img);

    char ascii_symbols[CHARACTER_COUNT];
    // Create array of symbols
    for (int i = 0; i < CHARACTER_COUNT; i++) {
        ascii_symbols[i] = (char) characters_Arial[i].codePoint;
    }

    // Save symbols to GPU memory as constants
    cl_mem gpu_ascii = clCreateBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(char) * CHARACTER_COUNT, ascii_symbols, &error);
    checkError(error);

    // Save intensities to GPU memory as constants
    cl_mem gpu_intensities = clCreateBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(int32_t) * CHARACTER_COUNT, intensities, &error);
    checkError(error);

	// Create a program from source
	cl_program program = createProgram(loadKernel ("to_ascii.cl"),
		context);
    
    char defines_str[128];
    snprintf(defines_str, sizeof(defines_str),
        "-D LINE_HEIGHT=%d -D SYMBOL_WIDTH=%d -D CHARACTER_COUNT=%d -D LNWIDTH=1280",
        LINE_HEIGHT, SYMBOL_WIDTH, CHARACTER_COUNT);
	
    checkError(clBuildProgram (program, deviceIdCount, deviceIds.data (),
		defines_str, nullptr, nullptr));
    
    /*
    // Determine the size of the log
    size_t log_size;
    clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    // Allocate memory for the log
    char *log = (char *) malloc(log_size);

    // Get the log
    clGetProgramBuildInfo(program, deviceIds[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

    // Print the log
    printf("%s\n", log);
    */

    // Read the kernel code
	cl_kernel kernel = clCreateKernel(program, "calc_index", &error);
	checkError(error);


    // Create command queue
    cl_command_queue queue = clCreateCommandQueue(context, deviceIds [0],
        0, &error);
    checkError(error);

    
    // Setup the kernel arguments that don't change
    clSetKernelArg(kernel, 1, sizeof (cl_mem), &gpu_intensities);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &gpu_ascii);

    for (int i = 0; i < frm_count; i++) {
        // These arguments change with each call
        clSetKernelArg(kernel, 0, sizeof (cl_mem), &gpu_gray[i]);
        clSetKernelArg(kernel, 3, sizeof (cl_mem), &gpu_output[i]);

        // Run the frame processing
        std::size_t offset [3] = { 0 }; // can be just NULL, maybe will use it later
        std::size_t size [3] = { symbols_per_width, symbols_per_height, 1 };
        checkError(clEnqueueNDRangeKernel (queue, kernel, 2, offset, size, nullptr,
            0, nullptr, nullptr));
    }

	// Allocate memory for result on CPU
    int result_size = symbols_per_height*symbols_per_width*frm_count;
    char *ascii_img = new char[result_size];

	// Get the result back to the host
	std::size_t origin[3] = { 0 };
	std::size_t region[3] = { symbols_per_width, symbols_per_height, 1 };

    for (int i = 0; i < frm_count; i++) {
        clEnqueueReadBuffer (queue, gpu_output[i], CL_FALSE,
            0, sizeof(char)*symbols_per_height*symbols_per_width,
            &ascii_img[symbols_per_height*symbols_per_width*i], 0, nullptr, nullptr);
    }

    // Wait until all processing will be finished and
    // result copied back to host
    clFinish(queue);

    // Save result to file
    FILE *output = fopen("ascii_img.txt", "w");
    int frame_size = symbols_per_height * symbols_per_width;

    for (int f_numb = 0; f_numb < frm_count; f_numb++) {
        for (int y = 0; y < symbols_per_height; y++) {
            for (int x = 0; x < symbols_per_width; x++) {
                fputc(ascii_img[frame_size*f_numb + symbols_per_width*y + x], output);
            }
            fputc('\n', output);        
        }
        fputc('\n', output);
    }
    
    fclose(output);

    // Do cleanup
    for (int i = 0; i < frm_count; i++) {
        clReleaseMemObject(gpu_gray[i]);
        clReleaseMemObject(gpu_output[i]);
    }

    delete[] gpu_gray;
    delete[] gpu_output;
    delete[] gray_frames;
    delete[] ascii_img;

	clReleaseMemObject(gpu_intensities);
    clReleaseMemObject(gpu_ascii);

	clReleaseCommandQueue (queue);
	
	clReleaseKernel (kernel);
	clReleaseProgram (program);

	clReleaseContext (context);
}

void calc_font_intensities(int32_t intensities[], cv::Mat &charset) {
    for (int i = 0; i < CHARACTER_COUNT; i++) {
        Character c = characters_Arial[i];
        cv::Rect char_borders(c.x, c.y, c.width, c.height);
        cv::Mat c_img = charset(char_borders);
        intensities[i] = calculate_intensity(c_img);
    }
}

void get_character_set(cv::Mat &char_set) {
    char_set = cv::imread("font.png", CV_LOAD_IMAGE_GRAYSCALE);
    return;
}

int32_t calculate_intensity(cv::Mat img) {
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

std::string getPlatformName (cl_platform_id id)
{
	size_t size = 0;
	clGetPlatformInfo (id, CL_PLATFORM_NAME, 0, nullptr, &size);

	std::string result;
	result.resize (size);
	clGetPlatformInfo (id, CL_PLATFORM_NAME, size,
		const_cast<char*> (result.data ()), nullptr);

	return result;
}

void checkError (cl_int error)
{
	if (error != CL_SUCCESS) {
		std::cerr << "OpenCL call failed with error " << error << std::endl;
		std::exit (1);
	}
}

std::string getDeviceName (cl_device_id id)
{
	size_t size = 0;
	clGetDeviceInfo (id, CL_DEVICE_NAME, 0, nullptr, &size);

	std::string result;
	result.resize (size);
	clGetDeviceInfo (id, CL_DEVICE_NAME, size,
		const_cast<char*> (result.data ()), nullptr);

	return result;
}

std::string loadKernel (const char* name)
{
	std::ifstream in (name);
	std::string result (
		(std::istreambuf_iterator<char> (in)),
		std::istreambuf_iterator<char> ());
	return result;
}

cl_program createProgram (const std::string& source,
	cl_context context)
{
	size_t lengths [1] = { source.size () };
	const char* sources [1] = { source.data () };

	cl_int error = 0;
	cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
	checkError (error);

	return program;
}