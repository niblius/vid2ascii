int calculate_intensity(__global uchar* img, int2 pos, int WIDTH) {
    int intensity = 0;
    // use MAD punctions
    int linear_pos = pos.y*LINE_HEIGHT*LNWIDTH + SYMBOL_WIDTH*pos.x;

    for (int u = 0; u < LINE_HEIGHT; u++)
    {
        int u_pos = linear_pos + u*LNWIDTH;
        
        for (int v = 0; v < SYMBOL_WIDTH; v++)
        {
            intensity += 255 - img[u_pos + v];
        }
    }
    
    return intensity;
}

char fit_by_intensity(
    int intensity,
    __constant int* intens_pool,
    __constant char* ascii) {

    int diff, smallest_diff = abs_diff(intensity, intens_pool[0]);

    // Save index, not char, so that we don't touch __global
    int smallest_index = 0;
    for (int i = 1; i < CHARACTER_COUNT; i++) {
        diff = abs_diff(intensity, intens_pool[i]);
        if (diff < smallest_diff) {
            smallest_diff = diff;
            smallest_index = i;
        }
    }

    return ascii[smallest_index];
}

__kernel void calc_index (
    __global uchar* img,
	__constant int* intens_pool,
    __constant char* ascii,
	__global char* output)
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const int WIDTH = get_global_size(0);
    const int HEIGHT = get_global_size(1);

    int intensity = calculate_intensity(img, pos, WIDTH);
    char character = fit_by_intensity(intensity, intens_pool, ascii);
    output[WIDTH*pos.y + pos.x] = character;
}