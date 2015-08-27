#include "cudawrapper.h"

namespace L3D
{
    texture<float, 2, cudaReadModeElementType> tex_segments;
    texture<float4, 2, cudaReadModeElementType> tex_segments_f4;
    texture<float, 2, cudaReadModeElementType> tex_fundamentals;
    texture<float, 2, cudaReadModeElementType> tex_projections;
    texture<float, 2, cudaReadModeElementType> tex_RtKinv;
    texture<float, 2, cudaReadModeElementType> tex_centers;

    ////////////////////////////////////////////////////////////////////////////////
    // helper function for rounded-up division
    int divUp(int a, int b)
    {
        float res = float(a)/float(b);
        return ceil(res);
    }

    ////////////////////////////////////////////////////////////////////////////////
    cudaError_t bindTexture(texture<float, 2>& tex, L3D::DataArray<float>* mem)
    {
      tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
      tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
      tex.filterMode = cudaFilterModeLinear;
      tex.normalized = false;
      cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
      return cudaBindTexture2D( 0, &tex, mem->dataGPU(), &channel_desc,
                         mem->width(), mem->height(), mem->pitchGPU());
    }

    ////////////////////////////////////////////////////////////////////////////////
    cudaError_t bindTexture(texture<float4, 2>& tex, L3D::DataArray<float4>* mem)
    {
      tex.addressMode[0] = cudaAddressModeClamp; // Neumann Boundary Conditions
      tex.addressMode[1] = cudaAddressModeClamp; // Neumann Boundary Conditions
      tex.filterMode = cudaFilterModeLinear;
      tex.normalized = false;
      cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
      return cudaBindTexture2D( 0, &tex, mem->dataGPU(), &channel_desc,
                         mem->width(), mem->height(), mem->pitchGPU());
    }

    /// DEVICE FUNCTIONS
    ////////////////////////////////////////////////////////////////////////////////
    __device__ float3 D_convert_f4_to_f3_3D(float4 X, const bool normalize)
    {
        if(normalize && fabs(X.w) > L3D_EPS_G)
        {
            X /= X.w;
        }

        return make_float3(X.x,X.y,X.z);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Note: point needs to be normalized! (--> p.z == 1)
    __device__ float D_distance_p2l_2D_f3(const float3 line, const float3 p)
    {
        return fabs((line.x*p.x+line.y*p.y+line.z)/sqrtf(line.x*line.x+line.y*line.y));
    }

    ////////////////////////////////////////////////////////////////////////////////
    __device__ float D_distance_p2plane_3D_f3(const float3 n_plane, const float3 p_plane,
                                              const float3 p)
    {
        return fabs(dot(n_plane,p_plane-p));
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Note: directional/normal vectors should be normalized!
    __device__ float4 D_intersect_line_and_plane(const float3 n_plane, const float3 p_plane,
                                                 const float3 dir_line, const float3 p_line)
    {
        float4 result = make_float4(0,0,0,0);

        // check precision
        if(fabs(dot(dir_line,n_plane)) < L3D_EPS_G)
            return result;

        // compute intersection
       float t = (dot(p_plane,n_plane) - dot(n_plane,p_line)) / (dot(n_plane,dir_line));
       float3 intersection = p_line + t * dir_line;

       result.x = intersection.x;
       result.y = intersection.y;
       result.z = intersection.z;
       result.w = 1;

       return result;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Note: points needs to be normalized! (--> p.z == 1)
    __device__ float D_segment_length_2D_f3(const float3 p1, const float3 p2)
    {
        float3 v = p1-p2;
        return sqrtf(v.x*v.x+v.y*v.y);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Note: points needs to be normalized! (--> p.z == 1)
    __device__ float D_angle_between_lines_deg_2D_f3(const float3 p1, const float3 p2,
                                                     const float3 q1, const float3 q2)
    {
        float2 v1 = normalize(make_float2(p1.x-p2.x,p1.y-p2.y));
        float2 v2 = normalize(make_float2(q1.x-q2.x,q1.y-q2.y));

        float angle = acos(fmax(fmin(dot(v1,v2),1.0f),-1.0f))/CUDART_PI*180.0f;

        if(angle > 90.0f)
            angle = 180.0f-angle;

        return angle;
    }

    ////////////////////////////////////////////////////////////////////////////////
    __device__ float D_angle_between_lines_deg_3D_f3(const float3 P1, const float3 P2,
                                                     const float3 Q1, const float3 Q2)
    {
        float3 v1 = normalize(P1-P2);
        float3 v2 = normalize(Q1-Q2);

        float angle = acos(fmax(fmin(dot(v1,v2),1.0f),-1.0f))/CUDART_PI*180.0f;

        if(angle > 90.0f)
            angle = 180.0f-angle;

        return angle;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Note: points needs to be normalized! (--> p.z == 1),
    // q needs to be collinear with p1 and p2!
    __device__ bool D_point_on_segment_2D_f3(const float3 p1, const float3 p2,
                                             const float3 q)
    {
        float2 v1 = make_float2(p1.x-q.x,p1.y-q.y);
        float2 v2 = make_float2(p2.x-q.x,p2.y-q.y);
        return (dot(v1,v2) < L3D_EPS_G);
    }

    ////////////////////////////////////////////////////////////////////////////////
    __device__ float3 D_epipolar_line(const float3 p, const int camID,
                                      const bool transpose)
    {
        float _p[3],_l[3];
        _p[0] = p.x; _p[1] = p.y; _p[2] = p.z;
        _l[0] = 0.0f; _l[1] = 0.0f; _l[2] = 0.0f;

        for(int r=0; r<3; ++r)
        {
            for(int c=0; c<3; ++c)
            {
                if(!transpose)
                    _l[r] += tex2D(tex_fundamentals,c+0.5f,float(camID*3+r)+0.5f)*_p[c];
                else
                    _l[r] += tex2D(tex_fundamentals,r+0.5f,float(camID*3+c)+0.5f)*_p[c];
            }
        }

        return make_float3(_l[0],_l[1],_l[2]);
    }

    ////////////////////////////////////////////////////////////////////////////////
    __device__ float D_segment_overlap_2D(const float3 src_p1, const float3 src_p2,
                                          const float3 q1, const float3 q2)
    {
        /*
        float len_src = D_segment_length_2D_f3(src_p1,src_p2);
        float len_tgt = D_segment_length_2D_f3(q1,q2);

        if(len_src < 1.0f || len_tgt < 1.0f)
            return 0.0f;

        if(D_point_on_segment_2D_f3(src_p1,src_p2,q1) &&
           D_point_on_segment_2D_f3(src_p1,src_p2,q2))
        {
            // both target points within the ref segment
            return len_tgt/len_src;
        }
        else if(D_point_on_segment_2D_f3(q1,q2,src_p1) &&
                D_point_on_segment_2D_f3(q1,q2,src_p2))
        {
            // both source points within the tgt segment
            return 1.0f;
        }
        else if(D_point_on_segment_2D_f3(src_p1,src_p2,q1))
        {
            // overlap exists
            if(D_point_on_segment_2D_f3(q1,q2,src_p1))
                return D_segment_length_2D_f3(q1,src_p1)/len_src;
            else
                return D_segment_length_2D_f3(q1,src_p2)/len_src;
        }
        else if(D_point_on_segment_2D_f3(src_p1,src_p2,q2))
        {
            // overlap exists
            if(D_point_on_segment_2D_f3(q1,q2,src_p2))
                return D_segment_length_2D_f3(q2,src_p2)/len_src;
            else
                return D_segment_length_2D_f3(q2,src_p1)/len_src;
        }

        // no overlap
        return 0.0f;
        */

        float len_src = D_segment_length_2D_f3(src_p1,src_p2);
        float len_tgt = D_segment_length_2D_f3(q1,q2);

        if(len_src < 1.0f || len_tgt < 1.0f)
            return 0.0f;

        if(D_point_on_segment_2D_f3(src_p1,src_p2,q1) &&
           D_point_on_segment_2D_f3(src_p1,src_p2,q2))
        {
            // both target points within the ref segment
            return len_tgt/len_src;
        }
        else if(D_point_on_segment_2D_f3(q1,q2,src_p1) &&
                D_point_on_segment_2D_f3(q1,q2,src_p2))
        {
            // both source points within the tgt segment
            return len_src/len_tgt;
        }
        else if(D_point_on_segment_2D_f3(src_p1,src_p2,q1))
        {
            float len1 = D_segment_length_2D_f3(src_p2,q2);
            float len2 = D_segment_length_2D_f3(src_p1,q2);

            // overlap exists
            if(D_point_on_segment_2D_f3(q1,q2,src_p1) && len1 > L3D_EPS_G)
                return D_segment_length_2D_f3(q1,src_p1)/len1;
            else if(len2 > L3D_EPS_G)
                return D_segment_length_2D_f3(q1,src_p2)/len2;
        }
        else if(D_point_on_segment_2D_f3(src_p1,src_p2,q2))
        {
            float len1 = D_segment_length_2D_f3(src_p1,q1);
            float len2 = D_segment_length_2D_f3(src_p2,q1);

            // overlap exists
            if(D_point_on_segment_2D_f3(q1,q2,src_p2) && len1 > L3D_EPS_G)
                return D_segment_length_2D_f3(q2,src_p2)/len1;
            else if(len2 > L3D_EPS_G)
                return D_segment_length_2D_f3(q2,src_p1)/len2;
        }

        // no overlap
        return 0.0f;
    }

    ////////////////////////////////////////////////////////////////////////////////
    __device__ float3 normalize_hom_coords_2D(float3 p)
    {
        if(fabs(p.z) > L3D_EPS_G)
        {
            p /= p.z;
            p.z = 1;
            return p;
        }
        else
        {
            return make_float3(0,0,0);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    __device__ float3 D_get_ray_src(const float3 p, const float* RtKinv, const int stride)
    {
        float _p[3],_ray[3];
        _p[0] = p.x; _p[1] = p.y; _p[2] = p.z;
        _ray[0] = 0.0f; _ray[1] = 0.0f; _ray[2] = 0.0f;

        for(int r=0; r<3; ++r)
        {
            for(int c=0; c<3; ++c)
            {
                _ray[r] += RtKinv[r*stride+c]*_p[c];
            }
        }

        return make_float3(_ray[0],_ray[1],_ray[2]);
    }

    ////////////////////////////////////////////////////////////////////////////////
    __device__ float3 D_get_ray_tgt(const float3 p, const int cID)
    {
        float _p[3],_ray[3];
        _p[0] = p.x; _p[1] = p.y; _p[2] = p.z;
        _ray[0] = 0.0f; _ray[1] = 0.0f; _ray[2] = 0.0f;

        for(int r=0; r<3; ++r)
        {
            for(int c=0; c<3; ++c)
            {
                _ray[r] += tex2D(tex_RtKinv,c+0.5f,float(cID*3+r)+0.5f)*_p[c];
            }
        }

        return make_float3(_ray[0],_ray[1],_ray[2]);
    }

    ////////////////////////////////////////////////////////////////////////////////
    __device__ float D_get_triangulation_depth(const float3 p1, const float3 p2,
                                               const float3 C1, const float3 C2,
                                               const int camID2, const bool for_src,
                                               const float* RtKinv1, const int r_stride)
    {
        float3 ray1 = normalize(D_get_ray_src(p1,RtKinv1,r_stride));
        float3 ray2 = normalize(D_get_ray_tgt(p2,camID2));
        float3 w0 = C1-C2;

        float a = dot(ray1,ray1);
        float b = dot(ray1,ray2);
        float c = dot(ray2,ray2);
        float d = dot(ray1,w0);
        float e = dot(ray2,w0);

        float denom = a*c-b*b;
        if(fabs(denom) > L3D_EPS_G)
        {
            // triangulation possible
            if(for_src)
                return (b*e-c*d)/denom;
            else
                return (a*e-b*d)/denom;
        }
        else
        {
            // impossible correspondence
            return -1.0f;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    __device__ float3 D_unproject_point_src(const float3 p, const float3 C,
                                            const float depth,
                                            const float* RtKinv, const int r_stride)
    {
        float3 ray = normalize(D_get_ray_src(p,RtKinv,r_stride));
        return C+depth*ray;
    }

    ////////////////////////////////////////////////////////////////////////////////
    __device__ float3 D_unproject_point_tgt(const float3 p, const float3 C,
                                            const float depth, const int cID)
    {
        float3 ray = normalize(D_get_ray_tgt(p,cID));
        return C+depth*ray;
    }

    ////////////////////////////////////////////////////////////////////////////////
    __device__ float3 D_project_point_tgt(const float3 P, const int camID)
    {
        float _P[4];
        _P[0] = P.x; _P[1] = P.y; _P[2] = P.z; _P[3] = 1.0f;
        float _p[3];
        _p[0] = 0.0f; _p[1] = 0.0f; _p[2] = 0.0f;
        for(int r=0; r<3; ++r)
        {
            for(int c=0; c<4; ++c)
            {
                _p[r] += tex2D(tex_projections,c+0.5f,float(camID*3+r)+0.5f)*_P[c];
            }
        }

        if(fabs(_p[2]) > L3D_EPS_G)
        {
           return make_float3(_p[0]/_p[2],_p[1]/_p[2],1.0f);
        }
        else
        {
            return make_float3(0,0,0);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    __device__ float D_hypothesis_confidence(const float3 p1, const float3 p2,
                                             const float3 P1, const float3 P2,
                                             const float3 Q1, const float3 Q2,
                                             const float3 C, const int tgtID,
                                             const float sigma_p, const float sigma_a,
                                             const float spatial_k, const float median_depth)
    {
        // check 3D distances
        if(spatial_k > 0.0f && median_depth > 0.0f)
        {
            float depth1 = length(C-P1);
            float depth2 = length(C-P2);

            float unc1,unc2;

            if(depth1 < median_depth)
                unc1 = spatial_k*depth1;
            else
                unc1 = spatial_k*median_depth;

            if(depth2 < median_depth)
                unc2 = spatial_k*depth2;
            else
                unc2 = spatial_k*median_depth;

            float dist1 = length(P1-Q1);
            float dist2 = length(P2-Q2);

            if(dist1 > unc1 || dist2 > unc2)
                return 0.0f;
        }

        // src line
        float3 line1 = cross(p1,p2);

        // tgt data
        float4 data = tex2D(tex_segments_f4,tgtID+0.5f,0.5f);
        float3 q1 = make_float3(data.x,data.y,1.0f);
        float3 q2 = make_float3(data.z,data.w,1.0f);
        float3 line2 = cross(q1,q2);

        // distances
        float d1 = fmax(D_distance_p2l_2D_f3(line2,p1),
                        D_distance_p2l_2D_f3(line2,p2));
        float d2 = fmax(D_distance_p2l_2D_f3(line1,q1),
                        D_distance_p2l_2D_f3(line1,q2));
        float dist = fmax(d1,d2);

        // angle
        float angle = D_angle_between_lines_deg_3D_f3(P1,P2,Q1,Q2);
        float sigma_sqr_a = sigma_a*sigma_a;

        float sigma_sqr_d = sigma_p*sigma_p;
        float d = expf(-dist*dist/(2.0f*sigma_sqr_d));

        return fmin(d,expf(-angle*angle/(2.0f*sigma_sqr_a)));
    }

    /// KERNEL FUNCTIONS
    ////////////////////////////////////////////////////////////////////////////////
    __global__ void K_pairwise_line_segment_aff(float* affinities, const int size,
                                                const float sigma,
                                                const int stride)
    {
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x < size && y < size)
        {
            if(x == y)
            {
                // same segment --> no similarity
                affinities[y*stride+x] = 0.0f;
            }
            else if(x < y)
            {
                // line1
                float3 p1 = make_float3(tex2D(tex_segments,0.5f,x+0.5f),
                                        tex2D(tex_segments,1.5f,x+0.5f),1.0f);
                float3 p2 = make_float3(tex2D(tex_segments,2.5f,x+0.5f),
                                        tex2D(tex_segments,3.5f,x+0.5f),1.0f);
                float3 line1 = cross(p1,p2);

                // line2
                float3 q1 = make_float3(tex2D(tex_segments,0.5f,y+0.5f),
                                        tex2D(tex_segments,1.5f,y+0.5f),1.0f);
                float3 q2 = make_float3(tex2D(tex_segments,2.5f,y+0.5f),
                                        tex2D(tex_segments,3.5f,y+0.5f),1.0f);
                float3 line2 = cross(q1,q2);

                // distances
                float d1 = fmax(D_distance_p2l_2D_f3(line2,p1),D_distance_p2l_2D_f3(line2,p2));
                float d2 = fmax(D_distance_p2l_2D_f3(line1,q1),D_distance_p2l_2D_f3(line1,q2));
                float d = fmax(d1,d2);

                // affinity
                float aff = expf(-d*d/(2.0f*sigma));

                affinities[y*stride+x] = aff;
                affinities[x*stride+y] = aff;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    __global__ void K_collinearity(float* relation, const int size,
                                   const float coll_sigma_sqr,
                                   const int stride)
    {
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x < size && y < size)
        {
            if(x == y)
            {
                // same segment --> no similarity
                relation[y*stride+x] = 0.0f;
            }
            else if(x < y)
            {
                float result = 0.0f;

                // line1
                float3 p1 = make_float3(tex2D(tex_segments,0.5f,x+0.5f),
                                        tex2D(tex_segments,1.5f,x+0.5f),1.0f);
                float3 p2 = make_float3(tex2D(tex_segments,2.5f,x+0.5f),
                                        tex2D(tex_segments,3.5f,x+0.5f),1.0f);
                float3 line1 = cross(p1,p2);

                // line2
                float3 q1 = make_float3(tex2D(tex_segments,0.5f,y+0.5f),
                                        tex2D(tex_segments,1.5f,y+0.5f),1.0f);
                float3 q2 = make_float3(tex2D(tex_segments,2.5f,y+0.5f),
                                        tex2D(tex_segments,3.5f,y+0.5f),1.0f);
                float3 line2 = cross(q1,q2);

                // distances
                float d1 = fmax(D_distance_p2l_2D_f3(line2,p1),D_distance_p2l_2D_f3(line2,p2));
                float d2 = fmax(D_distance_p2l_2D_f3(line1,q1),D_distance_p2l_2D_f3(line1,q2));
                float d = fmax(d1,d2);

                // affinity
                float aff = expf(-d*d/(2.0f*coll_sigma_sqr));
                if(aff > L3D_COLLIN_AFF_T_G)
                {
                    // check for conflict (overlap)
                    float2 _p1 = make_float2(p1.x,p1.y);
                    float2 _p2 = make_float2(p2.x,p2.y);
                    float2 _q1 = make_float2(q1.x,q1.y);
                    float2 _q2 = make_float2(q2.x,q2.y);
                    float pos1 = dot(_q1-_p1,_q2-_p1);
                    float pos2 = dot(_q1-_p2,_q2-_p2);
                    float pos3 = dot(_p1-_q1,_p2-_q1);
                    float pos4 = dot(_p1-_q2,_p2-_q2);

                    if(pos1 > -L3D_EPS_G && pos2 > -L3D_EPS_G && pos3 > -L3D_EPS_G && pos4 > -L3D_EPS_G)
                        result = aff;
                }

                relation[y*stride+x] = result;
                relation[x*stride+y] = result;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    __global__ void K_pairwise_matches(float4* buffer, const int width, const int height,
                                       const float* RtKinv, const int offset,
                                       const int cID, const float3 C_src, const int stride,
                                       const int r_stride)
    {
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x < width && y < height)
        {
            float4 result = make_float4(0,0,0,0);

            // line src
            float3 p1 = make_float3(tex2D(tex_segments,0.5f,y+0.5f),
                                    tex2D(tex_segments,1.5f,y+0.5f),1.0f);
            float3 p2 = make_float3(tex2D(tex_segments,2.5f,y+0.5f),
                                    tex2D(tex_segments,3.5f,y+0.5f),1.0f);
            float3 line1 = cross(p1,p2);

            // line tgt
            float4 data = tex2D(tex_segments_f4,float(offset+x)+0.5f,0.5f);
            float3 q1 = make_float3(data.x,data.y,1.0f);
            float3 q2 = make_float3(data.z,data.w,1.0f);
            float3 line2 = cross(q1,q2);

            // epipolar lines
            float3 epi_p1 = D_epipolar_line(p1,cID,false);
            float3 epi_p2 = D_epipolar_line(p2,cID,false);
            float3 epi_q1 = D_epipolar_line(q1,cID,true);
            float3 epi_q2 = D_epipolar_line(q2,cID,true);

            // intersect
            float3 l2_p1 = normalize_hom_coords_2D(cross(line2,epi_p1));
            float3 l2_p2 = normalize_hom_coords_2D(cross(line2,epi_p2));
            float3 l1_q1 = normalize_hom_coords_2D(cross(line1,epi_q1));
            float3 l1_q2 = normalize_hom_coords_2D(cross(line1,epi_q2));

            if(int(l2_p1.z) == 0 || int(l2_p2.z) == 0 ||
                    int(l1_q1.z) == 0 || int(l1_q2.z) == 0)
            {
                // intersections not valid
                buffer[y*stride+x] = result;
                return;
            }

            // check if enough overlap
            float overlap1 = D_segment_overlap_2D(p1,p2,l1_q1,l1_q2);
            float overlap2 = D_segment_overlap_2D(q1,q2,l2_p1,l2_p2);

            if(fmin(overlap1,overlap2) > L3D_MIN_OVERLAP_LOWER_T_G &&
                    fmax(overlap1,overlap2) > L3D_MIN_OVERLAP_UPPER_T_G)
            {
                // potential match --> triangulate
                float3 C_tgt = make_float3(tex2D(tex_centers,0.5f,cID+0.5f),
                                           tex2D(tex_centers,1.5f,cID+0.5f),
                                           tex2D(tex_centers,2.5f,cID+0.5f));
                result.x = D_get_triangulation_depth(p1,l2_p1,C_src,C_tgt,
                                                     cID,true,RtKinv,r_stride);
                result.y = D_get_triangulation_depth(p2,l2_p2,C_src,C_tgt,
                                                     cID,true,RtKinv,r_stride);
                result.z = D_get_triangulation_depth(l1_q1,q1,C_src,C_tgt,
                                                     cID,false,RtKinv,r_stride);
                result.w = D_get_triangulation_depth(l1_q2,q2,C_src,C_tgt,
                                                     cID,false,RtKinv,r_stride);

                buffer[y*stride+x] = result;
            }
            else
            {
                // no match
                buffer[y*stride+x] = result;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    __global__ void K_verify_matches(float* matches, const int2* match_offsets,
                                     const int2* camera_offsets, const int size,
                                     const float* RtKinv, const float3 C_src,
                                     const float sigma_p, const float sigma_a,
                                     const float spatial_k, const float median_depth,
                                     const int stride, const int r_stride)
    {
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x == 0 && y < size)
        {
            // match data
            int srcID = matches[y*stride];
            int camID = matches[y*stride+1];

            // depth
            float depth_p1 = matches[y*stride+3];
            float depth_p2 = matches[y*stride+4];

            // segment data
            float3 p1 = make_float3(tex2D(tex_segments,0.5f,srcID+0.5f),
                                    tex2D(tex_segments,1.5f,srcID+0.5f),1.0f);
            float3 p2 = make_float3(tex2D(tex_segments,2.5f,srcID+0.5f),
                                    tex2D(tex_segments,3.5f,srcID+0.5f),1.0f);

            // unproject
            float3 P1 = D_unproject_point_src(p1,C_src,depth_p1,RtKinv,r_stride);
            float3 P2 = D_unproject_point_src(p2,C_src,depth_p2,RtKinv,r_stride);

            // iterate over matches
            int start = match_offsets[srcID].x;
            int end = start+match_offsets[srcID].y;

            float confidence = 0.0f;

            int current_cam = -1;
            float current_confidence = 0.0f;

            for(int i=start; i<end; ++i)
            {
                if(i == y)
                    continue;

                // match data
                int camID2 = matches[i*stride+1];
                int tgtID2 = matches[i*stride+2];
                int camFeatureOffset = camera_offsets[camID2].x;

                // unproject
                float depth_q1 = matches[i*stride+3];
                float depth_q2 = matches[i*stride+4];
                float3 Q1 = D_unproject_point_src(p1,C_src,depth_q1,RtKinv,r_stride);
                float3 Q2 = D_unproject_point_src(p2,C_src,depth_q2,RtKinv,r_stride);

                if(camID2 == camID)
                    continue;

                if(camID2 != current_cam)
                {
                    // update score
                    if(current_cam != -1)
                    {
                        confidence += current_confidence;
                    }

                    current_confidence = 0.0f;
                    current_cam = camID2;
                }

                // 2D confidence
                float3 proj1 = D_project_point_tgt(P1,camID2);
                float3 proj2 = D_project_point_tgt(P2,camID2);

                if(int(proj1.z) == 1 && int(proj2.z) == 1)
                {
                    float conf = D_hypothesis_confidence(proj1,proj2,P1,P2,Q1,Q2,
                                                         C_src,tgtID2+camFeatureOffset,
                                                         sigma_p,sigma_a,
                                                         spatial_k,median_depth);

                    if(conf > 0.5f)
                    {
                        // confidence
                        if(conf > current_confidence)
                            current_confidence = conf;
                    }
                }
            }

            // update once more
            confidence += current_confidence;

            // store confidence
            matches[y*stride+7] = confidence;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    __global__ void K_sparseMat_row_normalization(float4* data, const int* start_indices,
                                                  const int num_rows, const int num_entries)
    {
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x == 0 && y < num_rows)
        {
            int start = start_indices[y];

            if(y >= 0)
            {
                // compute sum
                float sum = 0.0f;
                int i = start;
                while(i < num_entries)
                {
                    float4 e = data[i];
                    int row = e.x;

                    if(row != y)
                        break;

                    sum += e.z;
                    ++i;
                }

                // check for precision errors
                if(sum < L3D_EPS_G)
                    sum = L3D_EPS_G;

                // normalize
                i = start;
                while(i < num_entries)
                {
                    int row = data[i].x;

                    if(row != y)
                        break;

                    data[i].z /= sum;
                    ++i;
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    __global__ void K_sparseMat_diffusion_step(const float4* P, const float4* W,
                                               const int* P_rows, const int* W_cols,
                                               float4* P_prime, const int* P_prime_rows,
                                               const int num_entries)
    {
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x == 0 && y < num_entries)
        {
            // get data
            float4 data = P[y];

            // transpose
            int r = data.y;
            int c = data.x;

            // row[P]*col[W]
            float mul = 0.0f;
            int start_P = P_rows[r];
            int start_W = W_cols[c];
            while(start_P < num_entries && start_W < num_entries)
            {
                float4 d1 = P[start_P];
                float4 d2 = W[start_W];

                int row1 = d1.x;
                int col2 = d2.y;

                if(row1 != r || col2 != c)
                    break;

                mul += (d1.z*d2.z);
                ++start_P;
                ++start_W;
            }

            // multiply with transposed
            mul *= data.z;

            if(mul < L3D_EPS_G)
                mul = L3D_EPS_G;

            // store
            int s = P_prime_rows[r];
            bool found = false;
            while(s < num_entries && !found)
            {
                float4 dat = P_prime[s];
                int row = dat.x;
                int col = dat.y;

                if(row != r)
                    break;

                if(col == c)
                {
                    P_prime[s].z = mul;
                    found = true;
                }

                ++s;
            }
        }
    }

    /// EXTERNAL FUNCTIONS
    ////////////////////////////////////////////////////////////////////////////////
    void compute_collinearity(L3D::DataArray<float>* segments,
                              L3D::DataArray<float>* relation,
                              const float collin_s)
    {
        // init
        unsigned int block_size = L3D_CU_BLOCK_SIZE_C;
        unsigned int size = segments->height();

        // bind static texture
        bindTexture(tex_segments,segments);

        // compute affinities
        dim3 dimBlock = dim3(block_size,block_size);
        dim3 dimGrid = dim3(divUp(size, dimBlock.x),
                            divUp(size, dimBlock.y));

        L3D::K_collinearity <<< dimGrid, dimBlock >>> (relation->dataGPU(),size,
                                                       collin_s*collin_s,
                                                       relation->strideGPU());

        // unbind texture
        cudaUnbindTexture(tex_segments);
    }

    ////////////////////////////////////////////////////////////////////////////////
    void compute_pairwise_matches(L3D::DataArray<float>* segments_src,
                                  L3D::DataArray<float>* RtKinv_src,
                                  L3D::DataArray<float4>* segments_tgt,
                                  L3D::DataArray<float>* RtKinv_tgt,
                                  L3D::DataArray<float>* camCenters_tgt,
                                  const float3 camCenter_src,
                                  L3D::DataArray<float>* fundamentals,
                                  L3D::DataArray<float>* projections,
                                  L3D::DataArray<int2>* offsets,
                                  std::list<unsigned int>& toBeMatched,
                                  std::list<L3D::L3DMatchingPair>& matches,
                                  std::map<unsigned int,unsigned int>& local2global,
                                  const unsigned int maxSegments, const unsigned int vID,
                                  const float uncertainty_k_upper,
                                  const float uncertainty_k_lower,
                                  const float sigma_p, const float sigma_a,
                                  const bool verify3D, const float spatial_k,
                                  float& median_depth,
                                  const bool verbose, const std::string prefix)
    {
        if(toBeMatched.size() == 0)
            return;

        // init
        unsigned int block_size = L3D_CU_BLOCK_SIZE_C;
        unsigned int max_width = maxSegments;
        unsigned int height = segments_src->height();

        // bind static texture
        bindTexture(tex_segments,segments_src);
        bindTexture(tex_segments_f4,segments_tgt);
        bindTexture(tex_RtKinv,RtKinv_tgt);
        bindTexture(tex_centers,camCenters_tgt);
        bindTexture(tex_fundamentals,fundamentals);
        bindTexture(tex_projections,projections);

        // init buffer
        L3D::DataArray<float4>* buffer = new L3D::DataArray<float4>(max_width,height,true);

        // compute matches
        dim3 dimBlock = dim3(block_size,block_size);
        dim3 dimGrid;

        std::list<unsigned int>::iterator it = toBeMatched.begin();
        for(; it!=toBeMatched.end(); ++it)
        {
            unsigned int localID = *it;

            if(verbose)
                std::cout << prefix << "[" << vID << "] <--> [" << local2global[localID] << "]" << std::endl;

            // setup grid
            unsigned int feature_offset = offsets->dataCPU(localID,0)[0].x;
            unsigned int width = offsets->dataCPU(localID,0)[0].y;
            dimGrid = dim3(divUp(width, dimBlock.x),
                           divUp(height, dimBlock.y));

            // match segments
            L3D::K_pairwise_matches <<< dimGrid, dimBlock >>> (buffer->dataGPU(),
                                                               width,height,RtKinv_src->dataGPU(),
                                                               feature_offset,localID,
                                                               camCenter_src,
                                                               buffer->strideGPU(),
                                                               RtKinv_src->strideGPU());

            // download
            buffer->download();

            // store raw matches
            for(unsigned int i=0; i<height; ++i)
            {
                for(unsigned int j=0; j<width; ++j)
                {
                    float4 depths = buffer->dataCPU(j,i)[0];
                    if(depths.x > 0.0f && depths.y > 0.0f && depths.z > 0.0f && depths.w > 0.0f)
                    {
                        // potential match
                        L3D::L3DMatchingPair mp;
                        mp.segID1_ = i;
                        mp.segID2_ = j;
                        mp.camID2_ = localID;
                        mp.depths_ = depths;
                        mp.active_ = true;
                        mp.confidence_ = 0.0f;
                        matches.push_back(mp);
                    }
                }
            }
        }

        // cleanup
        delete buffer;

        // verify matches (sort first!)
        matches.sort(L3D::sortMatchingPairs);
        if(verbose)
            std::cout << prefix << "#raw_matches:          " << matches.size() << std::endl;

        if(matches.size() == 0)
            return;

        L3D::DataArray<float>* rawMatches = new L3D::DataArray<float>(8,matches.size());
        L3D::DataArray<int2>* matchOffset = new L3D::DataArray<int2>(height,1);
        matchOffset->setValue(make_int2(-1,-1));

        unsigned int current_seg = height;
        unsigned int num_matches = 0;
        unsigned int starting_pos = 0;
        unsigned int pos = 0;
        std::list<L3D::L3DMatchingPair>::iterator mit = matches.begin();
        for(; mit!=matches.end(); ++mit,++pos)
        {
            L3D::L3DMatchingPair mp = *mit;

            if(mp.segID1_ != current_seg)
            {
                // new segment begins
                if(current_seg != height)
                {
                    // update values
                    matchOffset->dataCPU(current_seg,0)[0] = make_int2(starting_pos,
                                                                       num_matches);

                    // reset
                    num_matches = 0;
                    starting_pos = pos;
                }

                current_seg = mp.segID1_;
            }

            rawMatches->dataCPU(0,pos)[0] = current_seg;
            rawMatches->dataCPU(1,pos)[0] = mp.camID2_;
            rawMatches->dataCPU(2,pos)[0] = mp.segID2_;
            rawMatches->dataCPU(3,pos)[0] = mp.depths_.x;
            rawMatches->dataCPU(4,pos)[0] = mp.depths_.y;
            rawMatches->dataCPU(5,pos)[0] = mp.depths_.z;
            rawMatches->dataCPU(6,pos)[0] = mp.depths_.w;
            rawMatches->dataCPU(7,pos)[0] = 0.0f;  // confidence

            ++num_matches;
        }

        if(current_seg < height && num_matches > 0)
        {
            matchOffset->dataCPU(current_seg,0)[0] = make_int2(starting_pos,
                                                               num_matches);
        }

        rawMatches->upload();
        matchOffset->upload();

        dimBlock = dim3(1,block_size*block_size);
        dimGrid = dim3(divUp(1, dimBlock.x),
                       divUp(rawMatches->height(), dimBlock.y));

        L3D::K_verify_matches <<< dimGrid, dimBlock >>> (rawMatches->dataGPU(),matchOffset->dataGPU(),
                                                         offsets->dataGPU(),rawMatches->height(),
                                                         RtKinv_src->dataGPU(),camCenter_src,
                                                         sigma_p,sigma_a,-1.0f,-1.0f,
                                                         rawMatches->strideGPU(),
                                                         RtKinv_src->strideGPU());

        // download
        matches.clear();
        rawMatches->download();

        std::vector<float> depths;
        float conf_t = 1.00f;
        unsigned int num_valid = 0;
        for(unsigned int i=0; i<matchOffset->width(); ++i)
        {
            int2 range = matchOffset->dataCPU(i,0)[0];

            int start = range.x;
            int end = start + range.y;

            if(start >= 0)
            {
                float max_conf = 0.0f;

                float depth_s1 = 0.0f;
                float depth_s2 = 0.0f;

                for(int k=start; k<end; ++k)
                {
                    float conf = rawMatches->dataCPU(7,k)[0];

                    if(conf > conf_t)
                        ++num_valid;

                    if(conf > max_conf)
                    {
                        max_conf = conf;

                        depth_s1 = rawMatches->dataCPU(3,k)[0];
                        depth_s2 = rawMatches->dataCPU(4,k)[0];
                    }
                }

                if(max_conf > conf_t/2.0f)
                {
                    depths.push_back(depth_s1);
                    depths.push_back(depth_s2);
                }
            }
        }

        median_depth = -1.0f;
        float median_reg_upper = 0.0f;
        float median_reg_lower = 0.0f;
        if(depths.size() > 0)
        {
            std::sort(depths.begin(),depths.end());
            median_depth = depths[depths.size()/2];

            median_reg_upper = median_depth*uncertainty_k_upper;
            median_reg_lower = median_depth*uncertainty_k_lower;
        }

        if(verbose)
            std::cout << prefix << "#filtered_matches (1): " << num_valid << std::endl;

        if(verbose)
            std::cout << prefix << "spatial_reg:           " << median_reg_lower << " - " << median_reg_upper << " (@depth: " << median_depth << ")" << std::endl;

        if(verify3D)
        {
            // verify confidences (3D)
            L3D::K_verify_matches <<< dimGrid, dimBlock >>> (rawMatches->dataGPU(),matchOffset->dataGPU(),
                                                             offsets->dataGPU(),rawMatches->height(),
                                                             RtKinv_src->dataGPU(),camCenter_src,
                                                             sigma_p,sigma_a,spatial_k,median_depth,
                                                             rawMatches->strideGPU(),
                                                             RtKinv_src->strideGPU());

            rawMatches->download();
        }

        rawMatches->removeFromGPU();
        matchOffset->removeFromGPU();

        // store result
        float confidence_norm = 2.0f;
        for(unsigned int i=0; i<rawMatches->height(); ++i)
        {
            float conf = rawMatches->dataCPU(7,i)[0];
            if(conf > conf_t)
            {
                conf /= confidence_norm;

                L3D::L3DMatchingPair mp;
                mp.segID1_ = rawMatches->dataCPU(0,i)[0];
                unsigned int locID = rawMatches->dataCPU(1,i)[0];
                mp.camID2_ = local2global[locID];
                mp.segID2_ = rawMatches->dataCPU(2,i)[0];
                mp.depths_ = make_float4(rawMatches->dataCPU(3,i)[0],
                                         rawMatches->dataCPU(4,i)[0],
                                         rawMatches->dataCPU(5,i)[0],
                                         rawMatches->dataCPU(6,i)[0]);
                mp.confidence_ = conf;
                mp.active_ = true;
                matches.push_back(mp);
            }
        }

        if(verbose)
        {
            std::cout << prefix << "#filtered_matches (2): " << matches.size() << std::endl;
        }

        // unbind textures
        cudaUnbindTexture(tex_segments);
        cudaUnbindTexture(tex_segments_f4);
        cudaUnbindTexture(tex_RtKinv);
        cudaUnbindTexture(tex_centers);
        cudaUnbindTexture(tex_fundamentals);
        cudaUnbindTexture(tex_projections);

        delete rawMatches;
        delete matchOffset;
    }

    ////////////////////////////////////////////////////////////////////////////////
    void replicator_dynamics_diffusion(L3D::SparseMatrix* &W, const bool verbose,
                                       const std::string prefix)
    {
        // init
        unsigned int block_size = L3D_CU_BLOCK_SIZE_C;
        unsigned int num_rows_cols = W->num_rows_cols();
        unsigned int num_entries = W->num_entries();
        dim3 dimBlock = dim3(1,block_size*block_size);
        dim3 dimGrid_RC = dim3(divUp(1, dimBlock.x),
                               divUp(num_rows_cols, dimBlock.y));
        dim3 dimGrid = dim3(divUp(1, dimBlock.x),
                            divUp(num_entries, dimBlock.y));

        // create P matrix
        L3D::SparseMatrix* P = new L3D::SparseMatrix(W,true);

        // make copy of P
        L3D::SparseMatrix* P_prime = new L3D::SparseMatrix(P);

        // row normalize
        L3D::K_sparseMat_row_normalization <<< dimGrid_RC, dimBlock >>> (P->entries()->dataGPU(),
                                                                         P->start_indices()->dataGPU(),
                                                                         num_rows_cols,num_entries);

        cudaDeviceSynchronize();

        for(int i=0; i<L3D_RDD_MAX_ITER; ++i)
        {
            // diffusion
            if(verbose)
                std::cout << prefix << "iteration: " << i << std::endl;

            // update
            L3D::K_sparseMat_diffusion_step <<< dimGrid, dimBlock >>> (P->entries()->dataGPU(),W->entries()->dataGPU(),
                                                                       P->start_indices()->dataGPU(),W->start_indices()->dataGPU(),
                                                                       P_prime->entries()->dataGPU(),P_prime->start_indices()->dataGPU(),
                                                                       num_entries);

            cudaDeviceSynchronize();

            // row normalize
            L3D::SparseMatrix* tmp = P;
            P = P_prime;
            P_prime = tmp;

            if(i < L3D_RDD_MAX_ITER-1)
            {
                L3D::K_sparseMat_row_normalization <<< dimGrid_RC, dimBlock >>> (P->entries()->dataGPU(),
                                                                                 P->start_indices()->dataGPU(),
                                                                                 num_rows_cols,num_entries);
            }

            cudaDeviceSynchronize();
        }

        // re-assign
        delete W;
        W = P;

        delete P_prime;
    }
}
