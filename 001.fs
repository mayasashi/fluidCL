#version 410 core

in vec2 uvout;
out vec4 color;

uniform sampler2D tex;

void main()
{
	color = texture(tex,uvout) + 0.0*vec4(0.0f,0.0f,1.0f,1.0f);
}