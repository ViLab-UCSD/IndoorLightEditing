/* 
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "helpers.h"
#include "structs/prd.h"
#include "random.h"
#include "commonStructs.h"
#include "lightStructs.h"
#include <vector>

using namespace optix;

rtDeclareVariable(float3,       cameraW, , );
rtDeclareVariable( float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable( float3, geometric_normal, attribute geometric_normal, );

rtDeclareVariable( float, t_hit, rtIntersectionDistance, );

rtDeclareVariable(optix::Ray, ray,   rtCurrentRay, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow, rtPayload, );
rtDeclareVariable(float, scene_epsilon, , );

// Geometry Group
rtDeclareVariable( rtObject, top_object, , );

rtDeclareVariable(int, areaTriangleNum, , );
rtBuffer<areaLight> areaLights;
rtBuffer<float> areaLightCDF;
rtBuffer<float> areaLightPDF;

RT_CALLABLE_PROGRAM void sampleAreaLight(unsigned int& seed, float3& position ){
    float randf = rnd(seed);

    int left = 0, right = areaTriangleNum;
    int middle = int( (left + right) / 2);
    while(left < right){
        if(areaLightCDF[middle] <= randf)
            left = middle + 1;
        else if(areaLightCDF[middle] > randf)
            right = middle;
        middle = int( (left + right) / 2);
    }
    areaLight L = areaLights[left];
    
    float3 v1 = L.vertices[0];
    float3 v2 = L.vertices[1];
    float3 v3 = L.vertices[2];


    float ep1 = rnd(seed);
    float ep2 = rnd(seed);
    
    float u = 1 - sqrt(ep1);
    float v = ep2 * sqrt(ep1);

    position = v1 + (v2 - v1) * u + (v3 - v1) * v;
}


RT_PROGRAM void closest_hit_radiance()
{ 
    const float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
    const float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
    float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );  
    
    float3 hitPoint = ray.origin + t_hit * ray.direction;
    
    float3 position;
    sampleAreaLight(prd_radiance.seed, position );
    
    float3 N = ffnormal;
    if(dot(N, ray.direction ) > 0 )
        N = -N;

    float Dist = length(position - hitPoint );
    float3 L = normalize(position - hitPoint );

    Ray shadowRay = make_Ray(hitPoint + 0.2 * scene_epsilon * L, L, 1, scene_epsilon, Dist - scene_epsilon );
    PerRayData_shadow prd_shadow; 
    prd_shadow.inShadow = false;
    rtTrace(top_object, shadowRay, prd_shadow ); 
    if(prd_shadow.inShadow == false && dot(N, L) >= 0 ){ 
        prd_radiance.shadow = 1.0;
    }
    else{
        prd_radiance.shadow = 0.0;
    }
}


// any_hit_shadow program for every material include the lighting should be the same
RT_PROGRAM void any_hit_shadow()
{
    prd_shadow.inShadow = true;
    rtTerminateRay();
}

RT_PROGRAM void miss(){
    prd_radiance.shadow = -1.0;
}
