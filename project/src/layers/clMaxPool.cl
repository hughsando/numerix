
__kernel void MaxPool2x2(const __global float* src, __global float* dest, const int destW, const int destH, const int features, const int srcLastX, int const srcLastY, const int srcShift,const int srcStride) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int srcY = y<<srcShift;
    const int srcX = x<<srcShift;
    const int dy = srcY<srcLastY ? srcStride : 0;
    const int dx = srcX<srcLastX ? features : 0;

    const int o0 = srcY*srcStride + srcX*features;
    const int o1 = o0+dx;
    const int o2 = o0+dy;
    const int o3 = o2+dx;
    const int dOff = (y*destW+x)*features;
    for(int f=0;f<features;f++) {
       float m0 = src[o0+f];
       float m1 = src[o1+f];
       float ma = m0>m1 ? m0 : m1;
       m0 = src[o2+f];
       m1 = src[o3+f];
       float mb = m0>m1 ? m0 : m1;
       dest[dOff+f] = ma>mb ? ma:mb;
    }
}
;


__kernel void MaxPool3x3(const __global float* src, __global float* dest, const int destW, const int destH, const int features, const int srcLastX, int const srcLastY, const int srcShift,const int srcStride) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int srcX = x<<srcShift;
    const int srcY = y<<srcShift;

    //0 1 2
    //3 4 5
    //6 7 8
    const int o0 = srcY*srcStride + srcX*features;
    const int o1 = o0+features;
    const int o2 = srcX+1<srcLastX ? o1+features : o0;
    const int o3 = o0+srcStride;
    const int o4 = o1+srcStride;
    const int o5 = o2+srcStride;
    const int o6 = srcY+1<srcLastY ? o3 + srcStride : o3;
    const int o7 = srcY+1<srcLastY ? o4 + srcStride : o4;
    const int o8 = srcY+1<srcLastY ? o5 + srcStride : o5;


    /*
    //7 4 8
    //3 0 1
    //6 2 5
    const int o0 = srcY*srcStride + srcX*features;
    const int o1 = srcX<srcLastX ? o0+features : o0;
    const int o2 = srcY<srcLastY ? o0+srcStride : o0;
    const int o3 = srcX>0 ? o0-features : o0;
    const int o4 = srcY>0 ? o0-srcStride : o0;

    const int o5 = srcX<srcLastX ? o2+features : o0;
    const int o6 = srcX>0 ? o2-features : o0;
    const int o7 = srcX>0 ? o4-features : o0;
    const int o8 = srcY>0 ? o1-srcStride : o0;
    */

    const int dOff = (y*destW+x)*features;
    for(int f=0;f<features;f++) {
       float ma = src[o0];
       ma = max(ma,src[o1]);
       ma = max(ma,src[o2]);
       ma = max(ma,src[o2]);
       ma = max(ma,src[o3]);
       ma = max(ma,src[o4]);
       ma = max(ma,src[o5]);
       ma = max(ma,src[o6]);
       ma = max(ma,src[o7]);
       dest[dOff+f] = max(ma,src[o8]);
       src++;
    }
}
;

