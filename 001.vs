#version 410 core

layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 uv;

out vec2 uvout;

void main()
{
	uvout = uv;
	gl_Position = vec4(pos,1.0f);
}