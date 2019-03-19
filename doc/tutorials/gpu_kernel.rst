==========================
Running kernels on the GPU
==========================

We will see how we can run kernels on the GPU. The following is a typical CUDA
code which adds the elements of two arrays of a given size:

.. code-block:: c

   // file: add.h
   #ifdef __cplusplus
     extern "C" void add(int n, float* x, float* y, float* r);
   #else
     extern void add(int n, float* x, float* y, float* r);
   #endif

.. code-block:: c

   // file: add.cu
   
   #include "add.h"

   __global__
   void device_add(int n, float* x, float* y, float* r)
   {
       int index = blockIdx.x * blockDim.x + threadIdx.x;
       int stride = blockDim.x * gridDim.x;
       for (int i = index; i < n; i += stride)
           r[i] = x[i] + y[i];
   }

   void add(int n, float* x, float* y, float* r)
   {
       int blockSize = 256;
       int numBlocks = (n + blockSize - 1) / blockSize;
       device_add<<<numBlocks, blockSize>>>(n, x, y, r);
   }


The ``device_add`` function is called a CUDA kernel (not to be confused with the
``gumath`` kernels!). This is what will actually run on the GPU. The reason why
a GPU is faster than a CPU is because it can massively parallelize
computations, and this is why we have these ``index`` and ``stride`` variables:
the kernel will be applied on different parts of the data at the same time.

Our ``gumath`` kernel however will use the ``add`` function, which
internally calls ``device_add`` with a special CUDA syntax and some extra-parameters
(basically specifying how much it will be parallelized).


.. code-block:: bash


Now we need to generate the ``gumath`` kernel for our ``add`` function.
The corresponding configuration file looks like this:

.. code-block:: none

   # file: add-kernels.cfg

   [MODULE add]
   typemaps = 
   	float: float32
   	int: int32
   includes = 
   	add.h
   include_dirs = 
   sources =
   	add.cu
   	
   libraries = 
   	
   library_dirs = 
   	
   header_code = 
   kinds = C
   ellipses = none
   
   [KERNEL add]
   prototypes = 
   	void add(int   n, float *  x, float *  y, float *  r);
   description = 
   dimension = x(n), y(n), r(n)
   input_arguments = x, y
   inplace_arguments = r
   hide_arguments = n = len(x)

We can now generate the kernel:

.. code-block:: bash

   $ xnd_tools kernel add-kernels.cfg
   $ xnd_tools module add-kernels.cfg

And create a static library:

.. code-block:: bash

   $ nvcc --compiler-options '-fPIC' -c add.cu
   $ gcc -fPIC                                               \
     -c add-kernels.c                                        \
     -I$SITE_PACKAGES/xndtools/kernel_generator              \
     -I$SITE_PACKAGES/xnd                                    \
     -I$SITE_PACKAGES/ndtypes                                \
     -I$SITE_PACKAGES/gumath
   $ ar rcs libadd.a add.o add-kernels.o

Finally, launch ``python setup.py install`` with this ``setup.py`` file:

.. code-block:: python

   # file: setup.py

   from distutils.core import setup, Extension
   from distutils.sysconfig import get_python_lib
   
   site_packages = get_python_lib()
   lib_dirs = [f'{site_packages}/{i}' for i in ['ndtypes', 'gumath', 'xnd']]
   
   module1 = Extension('add',
                       include_dirs = lib_dirs,
                       libraries = ['add', 'ndtypes','gumath', 'xnd', 'cudart', 'stdc++'],
                       library_dirs = ['.', '/usr/local/cuda-9.2/lib64'] + lib_dirs,
                       sources = ['add-python.c'])
   
   setup (name = 'add',
          version = '1.0',
          description = 'This is a gumath kernel extension that adds two XND containers on the GPU',
          ext_modules = [module1])

If everything went fine, you should be able to run the kernel on the GPU::

   >>> import gumath
   >>> from xnd import *
   >>> from add import add
   >>>
   >>> x = xnd.empty("1048576 * float32", device="cuda:managed")
   >>> y = xnd.empty("1048576 * float32", device="cuda:managed")
   >>> out = xnd.empty("1048576 * float32", device="cuda:managed")
   >>>
   >>> for i in range(1048576):
   ...     x[i] = i
   ...     y[i] = 2 * i
   ...
   >>>
   >>> out
   xnd([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...], type='1048576 * float32')
   >>>
   >>> add(x, y, out)
   >>>
   >>> out
   xnd([0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, ...], type='1048576 * float32')
