
__kernel void MaxPool2x2(const __global float* src, __global float* dest, const int destW, const int destH, const int srcLastX, int const srcLastY, const int srcShift,const int srcStride) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int srcY = y<<srcShift;
    const int srcX = x<<srcShift;
    const int dy = srcY<srcLastY ? srcStride : 0;
    const int dx = srcX<srcLastX ? CHANNELS : 0;

    const int o0 = srcY*srcStride + srcX*CHANNELS;
    const int o1 = o0+dx;
    const int o2 = o0+dy;
    const int o3 = o2+dx;
    const int dOff = (y*destW+x)*CHANNELS;
    for(int f=0;f<CHANNELS;f++) {
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


__kernel void MaxPool3x3(const __global float* src, __global float* dest, const int destW, const int destH, const int srcLastX, int const srcLastY, const int srcShift,const int srcStride) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int srcX = x<<srcShift;
    const int srcY = y<<srcShift;

    //0 1 2
    //3 4 5
    //6 7 8
    const int dx = CHANNELS>>3;
    const int dy = srcStride>>3;
    const int o0 = srcY*dy + srcX*dx;
    const int o1 = o0+dx;
    const int o2 = srcX+1<srcLastX ? o1+dx : o0;
    const int o3 = o0+dy;
    const int o4 = o1+dy;
    const int o5 = o2+dy;
    const int o6 = srcY+1<srcLastY ? o3 + dy : o3;
    const int o7 = srcY+1<srcLastY ? o4 + dy : o4;
    const int o8 = srcY+1<srcLastY ? o5 + dy : o5;

    const __global float8* src8 = (const __global float8*)src;
    __global float8* dest8 = (__global float8*)(dest + (y*destW+x)*CHANNELS );

    for(int f=0;f<CHANNELS;f+=8) {
       float8 ma = src8[o0];
       ma = max(ma,src8[o1]);
       ma = max(ma,src8[o2]);
       ma = max(ma,src8[o2]);
       ma = max(ma,src8[o3]);
       ma = max(ma,src8[o4]);
       ma = max(ma,src8[o5]);
       ma = max(ma,src8[o6]);
       ma = max(ma,src8[o7]);
       *dest8++ = max(ma,src8[o8]);
       src8++;
    }
}
;

