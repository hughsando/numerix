#include <Tensor.h>
#include <hx/Thread.h>

#include <stdexcept>

#define LOW_SENTINEL  67
#define HIGH_SENTINEL 68

namespace numerix
{

// TensorData



TensorData::~TensorData()
{
   if (cpu)
      freeCpuAligned(cpu);
}

TensorData *TensorData::incRef()
{
   HxAtomicInc(&refCount);
   return this;
}

void TensorData::decRef()
{
   // TODO - atomic
   int now = HxAtomicDec(&refCount) - 1;
   if (now<=0)
      delete this;
}





u8 *TensorData::allocCpuAligned(int size)
{
   // Allocate 6 bytes at beginning, plus up to and extra 12 to ensure 16-byte alignment, and
   //  1 at the end
   // Add 3 and trunc to ensure multiple of 4
   u8 *alloc = new u8[ (size+6+12+1+3) & ~3];
   int offset16 = 16 - (((int)(size_t)alloc) & 0xf);
   if (offset16<6)
      offset16 += 16;

   u8 *cpu = alloc + offset16;
   cpu[-1] = LOW_SENTINEL;
   cpu[size] = HIGH_SENTINEL;

   cpu[-2] = offset16;
   *(unsigned int *)(cpu-6) = size;
   return cpu;
}

void TensorData::freeCpuAligned(void *inData)
{
   if (!inData)
      throw std::logic_error("freeCpuAligned - null pointer");

   u8 *cpu = (u8 *)inData;
   if (cpu[-1]!=LOW_SENTINEL)
      throw std::logic_error("TensorData underwrite");

   unsigned int length = *(int *)(cpu-6);
   if (cpu[length] != HIGH_SENTINEL)
      throw std::logic_error("TensorData overwrite");

   int align16 = cpu[-2];
   u8 *ptr = cpu - align16;
   delete [] ptr;
}


} // end namespace numerix

