#Comments in makefiles are started by "#" characters!

#Define the compiler directory. CUDA programs are compiled with nvcc, not gcc.
NVCC = /Developer/NVIDIA/CUDA-10.0/bin/nvcc

#This is the default target. It will run when we call "make" if any 
#of the files ImageUtil.h, ImageUtil.cpp, or kernel.cu have been 
#changed since we made nbody.
nbody: ImageUtil.h ImageUtil.cpp kernel.cu
	#nvcc works exactly like gcc.
	$(NVCC) -o nbody kernel.cu ImageUtil.cpp

#This is the "run" target. It will run when we call "make run".
#It depends on the target nbody, so it will make sure that nbody 
#has been built first.
run: nbody
	#make the "out" directory. If the directory already exists, 
	#this command fails. The "-" prefix says to keep going even if the command fails.
	-mkdir out
	#call the executable
	./nbody
	#create the .gif
	convert -delay 10 -loop 0 out/image*.bmp animation.gif

#Calling "make clean" will delete the executable, all of the bitmaps,
#and also the animation.gif.
clean:
	-rm nbody
	-rm -r out
	-rm animation.gif

