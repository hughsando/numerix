#ifndef DYNAMIC_LOAD_INCLUDED
#define DYNAMIC_LOAD_INCLUDED

struct DynamicLibrary
{
   void *handle;

   DynamicLibrary(const char *inName);

   void *load(const char *inFuncName);

   bool ok() { return handle; }
};


#define DynamicFunction0( LIB,API,RET, DEFAULT, NAME ) \
 RET NAME() { \
    typedef RET (API *f)(); \
    static f func=0; \
    if (!func) func = (f)LIB.load(#NAME); \
    if (!func) { return DEFAULT; } \
    return func(); \
 }

#define DynamicFunction1( LIB,API,RET, DEFAULT, NAME, A ) \
 RET NAME(A a) { \
    typedef RET (API *f)(A); \
    static f func=0; \
    if (!func) func = (f)LIB.load(#NAME); \
    if (!func) { return DEFAULT; } \
    return func(a); \
 }

#define DynamicFunction2Name( LIB,API,RET, DEFAULT, NAME, FUNC_NAME,  A, B ) \
 RET FUNC_NAME(A a, B b) { \
    typedef RET (API *f)(A,B); \
    static f func=0; \
    if (!func) func = (f)LIB.load(#NAME); \
    if (!func) { return DEFAULT; } \
    return func(a,b); \
 }

#define DynamicFunction2( LIB,API,RET, DEFAULT, NAME, A, B ) \
   DynamicFunction2Name(LIB,API,RET, DEFAULT, NAME, NAME, A, B )

#define DynamicFunction3( LIB,API,RET, DEFAULT, NAME, A, B, C ) \
 RET NAME(A a, B b, C c) { \
    typedef RET (API *f)(A,B,C); \
    static f func=0; \
    if (!func) func = (f)LIB.load(#NAME); \
    if (!func) { return DEFAULT; } \
    return func(a,b,c); \
 }

#define DynamicFunction4( LIB,API,RET, DEFAULT, NAME, A, B, C, D ) \
 RET NAME(A a, B b, C c, D d) { \
    typedef RET (API *f)(A,B,C, D); \
    static f func=0; \
    if (!func) func = (f)LIB.load(#NAME); \
    if (!func) { return DEFAULT; } \
    return func(a,b,c,d); \
 }


#define DynamicFunction5( LIB,API,RET, DEFAULT, NAME, A, B, C, D, E ) \
 RET NAME(A a, B b, C c, D d, E e) { \
    typedef RET (API *f)(A,B,C,D,E); \
    static f func=0; \
    if (!func) func = (f)LIB.load(#NAME); \
    if (!func) { return DEFAULT; } \
    return func(a,b,c,d,e); \
 }


#define DynamicFunction6( LIB,API,RET, DEFAULT, NAME, A, B, C, D, E, G ) \
 RET NAME(A a, B b, C c, D d, E e, G g) { \
    typedef RET (API *f)(A,B,C,D,E, G); \
    static f func=0; \
    if (!func) func = (f)LIB.load(#NAME); \
    if (!func) { return DEFAULT; } \
    return func(a,b,c,d,e,g); \
 }


#define DynamicFunction7( LIB,API,RET, DEFAULT, NAME, A, B, C, D, E, G, H ) \
 RET NAME(A a, B b, C c, D d, E e, G g,H h) { \
    typedef RET (API *f)(A,B,C,D,E,G,H); \
    static f func=0; \
    if (!func) func = (f)LIB.load(#NAME); \
    if (!func) { return DEFAULT; } \
    return func(a,b,c,d,e,g,h); \
 }

#define DynamicFunction8( LIB,API,RET, DEFAULT, NAME, A, B, C, D, E, G, H, I ) \
 RET NAME(A a, B b, C c, D d, E e, G g,H h,I i) { \
    typedef RET (API *f)(A,B,C,D,E,G,H,I); \
    static f func=0; \
    if (!func) func = (f)LIB.load(#NAME); \
    if (!func) { return DEFAULT; } \
    return func(a,b,c,d,e,g,h,i); \
 }


#define DynamicFunction9( LIB,API,RET, DEFAULT, NAME, A, B, C, D, E, G, H, I, J) \
 RET NAME(A a, B b, C c, D d, E e, G g,H h,I i,J j) { \
    typedef RET (API *f)(A,B,C,D,E,G,H,I,J); \
    static f func=0; \
    if (!func) func = (f)LIB.load(#NAME); \
    if (!func) { return DEFAULT; } \
    return func(a,b,c,d,e,g,h,i,j); \
 }


#define DynamicFunction10( LIB,API,RET, DEFAULT, NAME, A, B, C, D, E, G, H, I, J, K) \
 RET NAME(A a, B b, C c, D d, E e, G g,H h,I i,J j,K k) { \
    typedef RET (API *f)(A,B,C,D,E,G,H,I,J,K); \
    static f func=0; \
    if (!func) func = (f)LIB.load(#NAME); \
    if (!func) { return DEFAULT; } \
    return func(a,b,c,d,e,g,h,i,j,k); \
 }


#define DynamicFunction11( LIB,API,RET, DEFAULT, NAME, A, B, C, D, E, G, H, I, J, K, L) \
 RET NAME(A a, B b, C c, D d, E e, G g,H h,I i,J j,K k,L l) { \
    typedef RET (API *f)(A,B,C,D,E,G,H,I,J,K,L); \
    static f func=0; \
    if (!func) func = (f)LIB.load(#NAME); \
    if (!func) { return DEFAULT; } \
    return func(a,b,c,d,e,g,h,i,j,k,l); \
 }


#define DynamicFunction12( LIB,API,RET, DEFAULT, NAME, A, B, C, D, E, G, H, I, J, K, L, M) \
 RET NAME(A a, B b, C c, D d, E e, G g,H h,I i,J j,K k,L l,M m) { \
    typedef RET (API *f)(A,B,C,D,E,G,H,I,J,K,L,M); \
    static f func=0; \
    if (!func) func = (f)LIB.load(#NAME); \
    if (!func) { return DEFAULT; } \
    return func(a,b,c,d,e,g,h,i,j,k,l,m); \
 }


#define DynamicFunction13( LIB,API,RET, DEFAULT, NAME, A, B, C, D, E, G, H, I, J, K, L, M, N) \
 RET NAME(A a, B b, C c, D d, E e, G g,H h,I i,J j,K k,L l,M m,N n) { \
    typedef RET (API *f)(A,B,C,D,E,G,H,I,J,K,L,M,N); \
    static f func=0; \
    if (!func) func = (f)LIB.load(#NAME); \
    if (!func) { return DEFAULT; } \
    return func(a,b,c,d,e,g,h,i,j,k,l,m,n); \
 }




#endif
