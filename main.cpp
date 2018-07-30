#include <iostream>
#include <fstream>
#include <string>

#include "GLcommon.h"
#include "simpleCL.h"
#include "simpleCLheaderGenerate.h"

#include "fluidSettings.h"


extern void SimulateFluid(simpleCLhandler & cl,kernelHandler & hndl);

int print()
{
	printf("-----------------------------------------\n");
	printf("-----------------------------------------\n");
	printf("press any key to continue...\n\n");
	char p[10];
	fgets(p, 10, stdin);
	printf("\n-----------------------------------------\n");
	printf("EXIT\n");
	return 0;
}

int main(int argc, const char **argv)

{

	simpleCLhandler cl = simpleCL_init();

	kernelHandler krnlhndl;

	krnlhndl.addKernelProgramFile("..\\cudaproj\\src\\myK.cl", "fluid", 1);
	krnlhndl.loadProgramFile();
	krnlhndl.buildProgram(cl);
	krnlhndl.printProgramBuildInfo(cl);

	float d = 1.0f / F_RES;
	float dt = d / 5.0f;
	float Re = 3000;
	double _mousePosX = 0;
	double _mousePosY = 0;

	float mousePosX = 0;
	float mousePosY = 0;
	float mousePosX_previous = 0;
	float mousePosY_previous = 0;

	float mouse_U = 0;
	float mouse_V = 0;

	int mouseButtonFlg = 0;

	cl_mem P = clCreateBuffer(cl->mainContext, CL_MEM_READ_WRITE, sizeof(float) * F_RES * F_RES, NULL, NULL);
	cl_mem U = clCreateBuffer(cl->mainContext, CL_MEM_READ_WRITE, sizeof(float) * F_RES * F_RES, NULL, NULL);
	cl_mem V = clCreateBuffer(cl->mainContext, CL_MEM_READ_WRITE, sizeof(float) * F_RES * F_RES, NULL, NULL);
	cl_mem Fx = clCreateBuffer(cl->mainContext, CL_MEM_READ_WRITE, sizeof(float) * F_RES * F_RES, NULL, NULL);
	cl_mem Fy = clCreateBuffer(cl->mainContext, CL_MEM_READ_WRITE, sizeof(float) * F_RES * F_RES, NULL, NULL);

	cl_kernel forcekernel = clCreateKernel(krnlhndl.getProgram(1), "AddForce", NULL);

	cl_kernel fluidkernel = clCreateKernel(krnlhndl.getProgram(1), "kernel_main", NULL);

	size_t global_work_size[1] = { F_RES * F_RES };

	GLcommon gl;

	gl.createWindowandMakeContext(512, 512);

	float posData[4 * 3] = {
		-1.0f,-1.0f,0.0f,
		 1.0f,-1.0f,0.0f,
		 -1.0f,1.0f,0.0f,
		 1.0f,1.0f,0.0f
	};
	float uvData[4 * 2] = {
		0.0f,0.0f,
		1.0f,0.0f,
		0.0f,1.0f,
		1.0f,1.0f
	};

	gl.VAO_Create(1, "cambus");
	gl.VBO_Create(1, "vbo_pos");
	gl.VBO_Create(2, "vbo_uv");

	gl.VBO_StoreData(1, 3, 4, GL_FLOAT, GL_STATIC_DRAW, GL_FALSE, 0, sizeof(float) * 3 * 4, posData);
	gl.VBO_StoreData(2, 2, 4, GL_FLOAT, GL_STATIC_DRAW, GL_FALSE, 0, sizeof(float) * 2 * 4, uvData);

	gl.Program_Create(1, "program");
	gl.Shader_Create(1, "vertex", GL_VERTEX_SHADER, "..\\cudaproj\\src\\001.vs");
	gl.Shader_Create(2, "fragment", GL_FRAGMENT_SHADER, "..\\cudaproj\\src\\001.fs");
	gl.Program_AttachShader(1, 1);
	gl.Program_AttachShader(1, 2);
	gl.Shader_AddAttribLocation(1, "pos", 0);
	gl.Shader_AddAttribLocation(1, "uv", 1);
	gl.Program_LinkShader(1);

	gl.VAO_VertexAttribArray_Register(1, 1, 1, 0);
	gl.VAO_VertexAttribArray_Register(1, 2, 1, 1);

	float *colorTexHstPtr = (float *)malloc(sizeof(float *) * F_RES * F_RES);
	for (int j = 0; j < F_RES; j++)
	{
		for (int i = 0; i < F_RES; i++)
		{
			colorTexHstPtr[i + F_RES * j] = i*1.0f / F_RES + j*1.0f / F_RES;
		}
	}
	gl.Texture_Create(1, "pattern texture", F_RES, F_RES);
	gl.Texture_Store(1, colorTexHstPtr, GL_RGBA8, GL_UNSIGNED_BYTE, GL_LINEAR, GL_LINEAR);
	gl.Texture_Register(1, 1);

	gl.Program_USE(1, [&]() {
		glUniform1i(glGetUniformLocation(gl.getProgram(1), "tex"), 1);
	});

	gl.Draw([&]() {

		mousePosX_previous = mousePosX;
		mousePosY_previous = mousePosY;
		glfwGetCursorPos(gl.window, &_mousePosX, &_mousePosY);
		mousePosX = _mousePosX * F_RES*1.0f / 512;
		mousePosY = _mousePosY * F_RES*1.0f / 512;
		mousePosX = fmax(fmin(mousePosX, F_RES - 2.0), 1.0);
		mousePosY = fmax(fmin(mousePosY, F_RES - 2.0), 1.0);
		mouse_U = mousePosX - mousePosX_previous;
		mouse_V = mousePosY - mousePosY_previous;
		mouseButtonFlg = (glfwGetMouseButton(gl.window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
		printf("%f,%f\n", mousePosX, mousePosY);

		clSetKernelArg(forcekernel, 0, sizeof(cl_mem), &Fx);
		clSetKernelArg(forcekernel, 1, sizeof(cl_mem), &Fy);
		clSetKernelArg(forcekernel, 2, sizeof(cl_float), &mouse_U);
		clSetKernelArg(forcekernel, 3, sizeof(cl_float), &mouse_V);
		clSetKernelArg(forcekernel, 4, sizeof(cl_float), &mousePosX);
		clSetKernelArg(forcekernel, 5, sizeof(cl_float), &mousePosY);
		clSetKernelArg(forcekernel, 6, sizeof(cl_int), &mouseButtonFlg);
		clEnqueueNDRangeKernel(cl->mainQueue, forcekernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

		clSetKernelArg(fluidkernel, 0, sizeof(cl_mem), &P);
		clSetKernelArg(fluidkernel, 1, sizeof(cl_mem), &U);
		clSetKernelArg(fluidkernel, 2, sizeof(cl_mem), &V);
		clSetKernelArg(fluidkernel, 3, sizeof(cl_mem), &Fx);
		clSetKernelArg(fluidkernel, 4, sizeof(cl_mem), &Fy);
		clSetKernelArg(fluidkernel, 5, sizeof(cl_float), &d);
		clSetKernelArg(fluidkernel, 6, sizeof(cl_float), &dt);
		clSetKernelArg(fluidkernel, 7, sizeof(cl_float), &Re);
		clEnqueueNDRangeKernel(cl->mainQueue, fluidkernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

		clEnqueueReadBuffer(cl->mainQueue, P, CL_FALSE, 0, sizeof(float)*F_RES*F_RES, colorTexHstPtr, 0, NULL, NULL);
		
		gl.Texture_Rewrite(1, colorTexHstPtr);

		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		gl.Program_USE(1);

		gl.VAO_USE(1);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

		gl.VAO_Unbind();
		gl.Program_Unbind();
	});

	if (colorTexHstPtr != NULL) free(colorTexHstPtr);

	clReleaseMemObject(P);
	clReleaseMemObject(U);
	clReleaseMemObject(V);
	clReleaseMemObject(Fx);
	clReleaseMemObject(Fy);


	simpleCL_close(cl);

	return print();
}