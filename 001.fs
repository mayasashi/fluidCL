#version 410 core

in vec2 uvout;
out vec4 color;

uniform sampler2D tex;
uniform sampler2D subtex;

uniform float a1;
uniform float a2;
uniform float a3;
uniform float a4;
uniform float a5;
uniform float a6;
uniform float a7;
uniform float a8;

uniform vec2 f00;
uniform vec2 f10;
uniform vec2 f01;
uniform vec2 f11;


void main()
{
	float X = uvout.x;
	float Y = uvout.y;
	float X_D = ((a5-a8*Y)*(a3-X) - (a2-a8*X)*(a6-Y)) / ((a5-a8*Y)*(a1-a7*X) - (a2-a8*X)*(a4-a7*Y));
	float Y_D = ((a4-a7*Y)*(a3-X) - (a1-a7*X)*(a6-Y)) / ((a4-a7*Y)*(a2-a8*X) - (a1-a7*X)*(a5-a8*Y));
	color = 0.5f*texture(tex,vec2(X_D,Y_D))+0.5f*texture(subtex,uvout);
	if(distance(f00,vec2(X,Y)) < 0.05f)color += vec4(0.3f,0.8f,0.3f,0.0f);
	if(distance(f10,vec2(X,Y)) < 0.05f)color += vec4(0.2f,0.6f,0.7f,0.0f);
	if(distance(f01,vec2(X,Y)) < 0.05f)color += vec4(0.4f,0.4f,0.4f,0.0f);
	if(distance(f11,vec2(X,Y)) < 0.05f)color += vec4(0.6f,0.2f,0.5f,0.0f);
}