#include <iostream>
#include <fstream>
#include <string>

#include "GLcommon.h"
#include "simpleCL.h"
#include "simpleCLheaderGenerate.h"

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

	krnlhndl.addKernelProgram("..\\cudaproj\\myK.cl", "fluid");
	krnlhndl.loadProgramFile();
	krnlhndl.buildProgram(cl);
	krnlhndl.printProgramBuildInfo(cl);

	//cl_mem A = clCreateBuffer(cl->mainContext, CL_MEM_READ_WRITE, sizeof(float) * 512 * 512, NULL, NULL);
	//clEnqueueWriteBuffer(cl->mainQueue, A, false, 0, sizeof(float) * 512 * 512, colorTexHstPtr, 0, NULL, NULL);



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
	gl.Shader_Create(1, "vertex", GL_VERTEX_SHADER, "..\\cudaproj\\001.vs");
	gl.Shader_Create(2, "fragment", GL_FRAGMENT_SHADER, "..\\cudaproj\\001.fs");
	gl.Program_AttachShader(1, 1);
	gl.Program_AttachShader(1, 2);
	gl.Shader_AddAttribLocation(1, "pos", 0);
	gl.Shader_AddAttribLocation(1, "uv", 1);
	gl.Program_LinkShader(1);

	gl.VAO_VertexAttribArray_Register(1, 1, 1, 0);
	gl.VAO_VertexAttribArray_Register(1, 2, 1, 1);

	float *colorTexHstPtr = (float *)malloc(sizeof(float *) * 512 * 512);
	for (int j = 0; j < 512; j++)
	{
		for (int i = 0; i < 512; i++)
		{
			colorTexHstPtr[i + 512 * j] = i / 512.0f + j / 512.0f;
		}
	}
	gl.Texture_Create(1, "pattern texture", 512, 512);
	gl.Texture_Store(1, colorTexHstPtr, GL_RGBA8, GL_UNSIGNED_BYTE, GL_LINEAR, GL_LINEAR);
	gl.Texture_Register(1, 1);

	gl.Program_USE(1, [&]() {
		glUniform1i(glGetUniformLocation(gl.getProgram(1), "tex"), 1);
	});

	gl.Draw([&]() {

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


	simpleCL_close(cl);

	return print();
}