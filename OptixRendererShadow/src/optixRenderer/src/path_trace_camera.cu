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

using namespace optix;

// Camera parameters
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        cameraU, , );
rtDeclareVariable(float3,        cameraV, , );
rtDeclareVariable(float3,        cameraW, , );

// Missing color
rtDeclareVariable(float,        bad_color, , ); 

// Scene variables  
rtDeclareVariable(unsigned int, sample_num, , );
rtDeclareVariable(unsigned int, height, ,  );
rtDeclareVariable(unsigned int, width, ,  );


// Rendering related
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(uint2,         launch_index, rtLaunchIndex, ); 
rtDeclareVariable(unsigned int, initSeed, , );
rtBuffer<float, 2>               output_buffer;



RT_PROGRAM void pinhole_camera()
{
    size_t2 screen = make_size_t2(width, height );
    
    float2 inv_screen = 1.0f / make_float2(screen) * 2.f;
    float2 pixel = (make_float2(launch_index ) ) * inv_screen - 1.f;

    float2 jitter_scale = inv_screen;

    unsigned int seed = tea<32>( 
        ( (initSeed)*(screen.x*launch_index.y+launch_index.x) + initSeed ), 
        ( (screen.y * launch_index.x + launch_index.y) * initSeed ) );
    
    float shadow = 0.0;
    for(int j = 0; j < sample_num; j++ ){
        // Sample pixel using jittering
        float2 jitter = make_float2(rnd(seed) - 0.5, rnd(seed)-0.5 );
        float2 d = pixel + jitter*jitter_scale;

        float3 ray_direction = normalize(d.x*cameraU + d.y*cameraV + cameraW);
        
        // Initialze per-ray data
        PerRayData_radiance prd; 
        prd.seed = seed; 

        // Each iteration is a segment of the ray path.  The closest hit will
        // return new segments to be traced here.
        Ray ray(eye, ray_direction, 0, scene_epsilon );
        rtTrace(top_object, ray, prd );

        shadow += prd.shadow;
        seed = prd.seed;
    }
    shadow = shadow / sample_num; 
    size_t2 output_pixel = make_size_t2(launch_index.x, launch_index.y ); 
    output_buffer[output_pixel ] = shadow; 
}

RT_PROGRAM void exception(){
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
}
