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

//-----------------------------------------------------------------------------
//
// optixVox: a sample that renders a subset of the VOX file format from MagicaVoxel @ ephtracy.
// Demonstrates non-triangle geometry, and naive random path tracing.
//
//-----------------------------------------------------------------------------

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <cstdlib>
#include <cstring>
#include <iostream> 
#include <fstream>
#include <vector>
#include <cmath>
#include <assert.h>  
#include <algorithm> 
#include "utils/ptxPath.h"
#include "tinyobjloader/objLoader.h"
#include "tinyplyloader/plyLoader.h"
#include "lightStructs.h" 
#include <fstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace optix; 

const float PI = 3.1415926535;

void createContext( 
        Context& context,   
        unsigned sampleNum,
        unsigned height, unsigned width,
        float fov )
{
    // Set up context
    context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 1 );
    context->setStackSize( 600 );
    
    fov = fov / 180.0 * PI;
    float tanX = tan(fov * 0.5 );
    float tanY = tanX / width * height;    

    float3 eye = make_float3(0.0, 0.0, 0.0 );
    float3 cameraU = make_float3(tanX, 0.0, 0.0);
    float3 cameraV = make_float3(0.0, tanY, 0.0);
    float3 cameraW = make_float3(0.0, 0.0, -1.0);
    context["eye"] -> setFloat(eye );
    context["cameraU"] -> setFloat(cameraU );
    context["cameraV"] -> setFloat(cameraV );
    context["cameraW"] -> setFloat(cameraW );

    Buffer outputBuffer = context -> createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT,  width, height ); 

    context["output_buffer"]->set( outputBuffer ); 
    context["sample_num"] -> setUint(sampleNum ); 
    context["height"] -> setUint(height ); 
    context["width"] -> setUint(width );

    // Ray generation program 
    std::string ptx_path( ptxPath( "path_trace_camera.cu" ) );
    Program ray_gen_program = context->createProgramFromPTXFile( ptx_path, "pinhole_camera" );
    context->setRayGenerationProgram( 0, ray_gen_program );  
    
    // Exception program
    Program exception_program = context->createProgramFromPTXFile( ptx_path, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( -1.0 );

    // Missing Program 
    std::string miss_path( ptxPath( "shadow.cu" ) );
    Program miss_program = context->createProgramFromPTXFile(miss_path, "miss");
    context->setMissProgram(0, miss_program);
}   

float computeDist(std::vector<float>& vertices, float ind1, float ind2 ){
    float z1 = vertices[3*ind1 + 2 ];
    float z2 = vertices[3*ind2 + 2 ]; 

    float dist = sqrt( (z1-z2)*(z1-z2) ); 
    float fac = std::max(std::min(-z1, -z2), float(1e-6 ) );
    dist /= fac  ; 
    return dist; 
} 


int createAreaLightsBuffer(
        Context& context, 
        const shape_t& shape
    )
{
    std::vector<float> cdfArr;
    std::vector<float> pdfArr;
    std::vector<areaLight> lights;
    float sum = 0;

    // Assign all the area lights 
    int faceNum = shape.mesh.indicesP.size() / 3;

    const std::vector<int>& faces = shape.mesh.indicesP;
    const std::vector<float>& vertices = shape.mesh.positions;

    for(int i = 0; i < faceNum; i++){
        int vId1 = faces[3*i], vId2 = faces[3*i+1], vId3 = faces[3*i+2]; 
        float3 v1 = make_float3(vertices[3*vId1], vertices[3*vId1+1], vertices[3*vId1+2]);
        float3 v2 = make_float3(vertices[3*vId2], vertices[3*vId2+1], vertices[3*vId2+2]);
        float3 v3 = make_float3(vertices[3*vId3], vertices[3*vId3+1], vertices[3*vId3+2]);
        
        float3 cproduct = cross(v2 - v1, v3 - v1);
        float area = 0.5 * sqrt(dot(cproduct, cproduct) ); 

        sum += area;
        cdfArr.push_back(sum);
        pdfArr.push_back(area );
        
        areaLight al;
        al.vertices[0] = v1;
        al.vertices[1] = v2;
        al.vertices[2] = v3;
        lights.push_back(al);
    } 

    // Computs the pdf and the cdf
    for(int i = 0; i < cdfArr.size(); i++){
        cdfArr[i] = cdfArr[i] / sum;
        pdfArr[i] = pdfArr[i] / sum;
    }

    Buffer lightBuffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_USER, lights.size() );
    lightBuffer->setElementSize( sizeof( areaLight) );
    memcpy( lightBuffer->map(), (char*)&lights[0], sizeof(areaLight) * lights.size() );
    lightBuffer->unmap(); 
    context["areaLights"]->set(lightBuffer );
    
    context["areaTriangleNum"] -> setInt(lights.size() ); 

    Buffer cdfBuffer = context -> createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, cdfArr.size() );
    memcpy(cdfBuffer -> map(), &cdfArr[0], sizeof(float) * cdfArr.size() );
    cdfBuffer -> unmap();
    
    Buffer pdfBuffer = context -> createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, pdfArr.size() );
    memcpy(pdfBuffer -> map(), &pdfArr[0], sizeof(float) * pdfArr.size() );
    pdfBuffer -> unmap();
    
    context["areaLightCDF"] -> set(cdfBuffer );
    context["areaLightPDF"] -> set(pdfBuffer );
}

void fromDepthToMesh(
        unsigned height, unsigned width, 
        float fov, float threshold, 
        const float* depth, 
        std::vector<float>& vertices, 
        std::vector<int>& faces )
{ 
    // Push vertices and vertex normals  
    fov = fov / 180.0 * PI;
    float tanX = tan(fov * 0.5 );
    float tanY = tanX / width * height;  
    float* x_c = new float[width * height ];
    float* y_c = new float[width * height ];
    for(int r = 0; r < height; r++){
        for(int c = 0; c < width; c++){
            float x = (float(c) / width * 2.0 - 1) * tanX; 
            float y = (1 - float(r) / height * 2.0) * tanY; 
            x_c[r * width + c ] = x;
            y_c[r * width + c ] = y; 
        }
    }
    
    for(int r = 0; r < height; r++){
        for(int c = 0; c < width; c++){
            int ind = r * width + c; 
            vertices.push_back(x_c[ind ] * depth[ind] );
            vertices.push_back(y_c[ind ] * depth[ind] );
            vertices.push_back(-depth[ind ] ); 
        }
    }

    // faces 
    for(int r = 0; r < height-1; r++ ){
        for(int c = 0; c < width-1; c++ ){
            int ind1 = r*width + c;
            int ind2 = r*width + c + 1;
            int ind3 = (r+1) * width + c; 
            int ind4 = (r+1) * width + c + 1; 
            
            float dist12 = computeDist(vertices, ind1, ind2);
            float dist24 = computeDist(vertices, ind2, ind4);
            float dist13 = computeDist(vertices, ind1, ind3);
            float dist34 = computeDist(vertices, ind3, ind4); 
            float dist14 = computeDist(vertices, ind1, ind4); 

            if( dist14 < (threshold * sqrt(2 ) ) )
            {
                if(dist12 < threshold && dist24 < threshold ){
                    faces.push_back(ind1 );
                    faces.push_back(ind4 );
                    faces.push_back(ind2 );
                } 

                if(dist13 < threshold && dist34 < threshold ){
                    faces.push_back(ind1 );
                    faces.push_back(ind3 );
                    faces.push_back(ind4 ); 
                }
            }
            else{
                float dist23 = computeDist(vertices, ind2, ind3);
                if(dist23 < (threshold * sqrt(2 ) ) )
                {
                    if(dist12 < threshold && dist13 < threshold ) {
                        faces.push_back(ind1 );
                        faces.push_back(ind3 );
                        faces.push_back(ind2 );
                    }
                    
                    if(dist24 < threshold && dist34 < threshold ){
                        faces.push_back(ind3 );
                        faces.push_back(ind4 );
                        faces.push_back(ind2 ); 
                    }
                }
            }

        }
    }

    delete [] x_c;
    delete [] y_c;
    return;
} 


void fillGeometry(
        std::vector<float>& vertices, 
        std::vector<int>& faces, 
        Geometry& geometry, 
        Context& context
        )
{
    int vertexNum = vertices.size() / 3;
    int faceNum = faces.size() / 3; 

    Buffer vertexBuffer = context -> createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, vertexNum );
    Buffer faceBuffer = context -> createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, faceNum );
    Buffer materialBuffer = context -> createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT, faceNum );
    
    float* vertexPt = reinterpret_cast<float*>(vertexBuffer -> map() );
    int* facePt = reinterpret_cast<int32_t*>(faceBuffer -> map() );
    int* materialPt = reinterpret_cast<int32_t*>( materialBuffer -> map() );

    for(int i = 0; i < vertexNum * 3; i++ ){
        vertexPt[i] = vertices[i];
    }
    for(int i = 0; i < faceNum * 3; i++ ){
        facePt[i] = faces[i];
    } 
    for(int i = 0; i < faceNum; i++ ){
        materialPt[i] = 0;
    }
    vertexBuffer -> unmap();
    faceBuffer -> unmap();
    materialBuffer -> unmap();
    
    geometry[ "vertex_buffer" ] -> setBuffer(vertexBuffer );
    geometry[ "index_buffer" ] -> setBuffer(faceBuffer );
    geometry[ "material_buffer"] -> setBuffer(materialBuffer );
    
    geometry -> setPrimitiveCount(faceNum );
} 

void fillMaterial(const std::string& matName, std::vector<Material>& optix_materials, Context& context )
{
    const std::string ptx_path = ptxPath(matName );
    Program ch_program = context->createProgramFromPTXFile( ptx_path, "closest_hit_radiance" );
    Program ah_program = context->createProgramFromPTXFile( ptx_path, "any_hit_shadow" );

    Material material = context->createMaterial();
    material->setClosestHitProgram( 0, ch_program );
    material->setAnyHitProgram( 1, ah_program );
    optix_materials.push_back(material );
}

void createGeometry(
        Context& context, 
        unsigned height, unsigned width, 
        float fov, float threshold, 
        const float* depth,  
        const std::string srcName, 
        const std::string fileName,  
        const std::string outputName )
{ 
    // Create geometry group;
    GeometryGroup geometry_group = context->createGeometryGroup();
    geometry_group->setAcceleration( context->createAcceleration( "Trbvh" ) ); 

    const std::string path = ptxPath( "triangle_mesh.cu" );
    optix::Program bounds_program = context->createProgramFromPTXFile( path, "mesh_bounds" );
    optix::Program intersection_program = context->createProgramFromPTXFile( path, "mesh_intersect" ); 
    
    // Build mesh 
    std::vector<float> vertices;
    std::vector<int> faces;    
    fromDepthToMesh( 
            height, width, fov, threshold, 
            depth, 
            vertices, faces );  

    int vertexNum = vertices.size() / 3;  
    int faceNum = faces.size() / 3;

    if(outputName != "None"){ 
        std::ofstream fOut(outputName.c_str(), std::ios::out ); 
        std::vector<float> vt;
        
        for(int h = 0; h < height; h++ ){
            for(int w = 0; w < width; w++){
                float tx = (w + 0.5) / float(width );
                float ty = (height - h - 0.5 ) / float(height );
                vt.push_back(tx );
                vt.push_back(ty );
            }
        }
        
        // Output vertices 
        for(int i = 0; i < vertexNum; i++){
            fOut<<"v "<<vertices[3 * i ]<<' ';
            fOut<<vertices[3 * i + 1]<<' ';
            fOut<<vertices[3 * i + 2]<<std::endl; 
            
            fOut<<"vt "<<vt[2 * i ]<<' ';
            fOut<<vt[2 * i + 1]<<std::endl;
        }

        // Output faces 
        for(int i = 0; i < faceNum; i++ ){
            fOut<<"f "<<faces[3*i] + 1<<'/'<<faces[3*i]+1<<' ';
            fOut<<faces[3*i+1] + 1<<'/'<<faces[3*i+1]+1<<' ';
            fOut<<faces[3*i+2] + 1<<'/'<<faces[3*i+2]+1<<std::endl;
        }

        fOut.close();

    }
    
    // Compute epsilon 
    float maxDist = 0;
    for(int i = 0; i < vertexNum; i++ ){
        float x = vertices[3* i ];
        float y = vertices[3* i + 1];
        float z = vertices[3* i + 2];
        float dist = sqrt(x * x + y * y + z * z );
        if(dist > maxDist )
            maxDist = dist; 
    }
    context["scene_epsilon"] -> setFloat(maxDist / 1e6 );
    
    // Push to program 
    Geometry geometry = context -> createGeometry();
    fillGeometry(vertices, faces, geometry, context );
    geometry->setBoundingBoxProgram(bounds_program );
    geometry->setIntersectionProgram(intersection_program );
    
    std::vector<Material> optix_materials; 
    fillMaterial("shadow.cu", optix_materials, context );
    GeometryInstance geom_instance = context->createGeometryInstance(
            geometry,
            optix_materials.begin(),
            optix_materials.end() );
    geometry_group -> addChild(geom_instance );
    
    shape_t shapeSrc; 
    std::string shapeType = srcName.substr(srcName.find_last_of(".") ); 
    bool isLoad = false;
    if(shapeType == std::string(".obj") ){  
        isLoad = objLoader::LoadObj(shapeSrc, srcName );
    }
    else if(shapeType == std::string(".ply") ){
        isLoad = plyLoader::LoadPly(shapeSrc, srcName );
    }
    else{
        std::cout<<"Unrecognize obj type"<<std::endl;
    }
    if(isLoad == false){
        std::cout<<"Fail to load light sources"<<std::endl;
    }     
    createAreaLightsBuffer(context, shapeSrc ); 

    // See if should insert a new object 
    if(fileName != std::string("None") ){
        shape_t shapeObj; 
        std::string shapeType = fileName.substr(fileName.find(".") ); 
        bool isLoad = false;
        if(shapeType == std::string(".obj") ){  
            isLoad = objLoader::LoadObj(shapeObj, fileName );
        }
        else if(shapeType == std::string(".ply") ){
            isLoad = plyLoader::LoadPly(shapeObj, fileName );
        }
        else{
            std::cout<<"Unrecognize obj type"<<std::endl;
        }
        if(isLoad == false){
            std::cout<<"Fail to load occluder"<<std::endl;
        }   

        // Push to program 
        Geometry geometry = context -> createGeometry(); 
        fillGeometry(shapeObj.mesh.positions, shapeObj.mesh.indicesP, geometry, context );
        
        geometry->setBoundingBoxProgram(bounds_program );
        geometry->setIntersectionProgram(intersection_program );
        
        std::vector<Material> optix_materials; 
        fillMaterial("shadow.cu", optix_materials, context );

        GeometryInstance geom_instance = context->createGeometryInstance(
                geometry,
                optix_materials.begin(),
                optix_materials.end() );
        geometry_group -> addChild(geom_instance );
    }
    context["top_object"] -> set(geometry_group );   

    return; 
} 

void getOutputBuffer(
    Context& context, 
    unsigned height, unsigned width, 
    float* output 
    )
{
    Buffer outputBuffer = context[ "output_buffer" ]->getBuffer();
    float* outputPt = reinterpret_cast<float*>(outputBuffer -> map() );  
    for(int r = 0; r < height; r++){
        for(int c = 0; c < width; c++){
            int ind1 = r * width + c;
            int ind2 = (height-1-r) * width + c;
            output[ind1] = outputPt[ind2 ];
        }
    }
    outputBuffer -> unmap(); 
    return;
} 


void renderShadows(
        const std::string& srcName, 
        unsigned sampleNum,
        unsigned height, unsigned width,  
        float fov, float threshold, 
        const float* depth, 
        float* output, 
        const std::string& fileName, 
        const std::string& outputName )
{ 
    Context context = 0; 
    createContext(context, 
            sampleNum, 
            height, width, 
            fov );  

    createGeometry(context, height, width, 
            fov, threshold, depth, srcName, fileName, outputName );
    
    srand(time(NULL) );
    context["initSeed"] -> setUint( rand() );
    context -> launch(0, width, height ); 
    
    getOutputBuffer(context, height, width, output );  
    
    context->destroy();
    context = 0;
    
    return;
}

namespace py = pybind11;

// int main( int argc, char** argv )
py::array_t<float>  py_render(
        py::array_t<float, py::array::c_style | py::array::forcecast> depth, int depth_h, int depth_w, 
        const std::string& srcName, 
        float fov, float threshold, int sampleNum, 
        const std::string& fileName, const std::string& outputName )
{
    bool use_pbo  = false;
    Context context = 0; 

    int depthSize [2];
    depthSize[0] = depth_h;
    depthSize[1] = depth_w;

    int total_num = depthSize[0] * depthSize[1]; 
    float* output = new float[total_num]; 

    renderShadows(
            srcName, 
            sampleNum,  
            depthSize[0], depthSize[1], 
            fov, threshold, 
            depth.data(), 
            output, 
            fileName, 
            outputName 
    );

    auto result        = py::array_t<float>(total_num);
    auto result_buffer = result.request();
    float *result_ptr    = (float *) result_buffer.ptr;

    std::memcpy(result_ptr, output, total_num * sizeof(float));

    return result;
}


PYBIND11_MODULE(optixRenderer,m)
{
    m.doc() = "optixRenderer";

    m.def("render", &py_render, "py render");
}


