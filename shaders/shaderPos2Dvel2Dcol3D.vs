#version 330 core

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aVel;
layout (location = 2) in vec3 aCol;
layout (location = 3) in int aMass;


out vec3 FragPos;
out vec3 FragVel;
out vec3 FragCol;  
out int FragMass;  

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
   gl_Position = projection * view * model * vec4(vec3(aPos.xy, 0.0f), 1.0f);
   FragPos = vec3(model * vec4(vec3(aPos.xy, 0.0f), 1.0f));
   FragVel = vec3(aVel.xy, 0.0f);
   FragCol = aCol;
   FragMass = aMass;
}
