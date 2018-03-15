// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

//Shows the grayscale of the depth from the camera.
 
Shader "Custom/DepthShader"
{
    SubShader
    {
        Pass
        {

            CGPROGRAM
            #pragma target 3.0
            #pragma vertex vert
            #pragma fragment frag
            #pragma debug

            #include "UnityCG.cginc"
 
            uniform sampler2D _CameraDepthTexture; //the depth texture
            sampler2D _MainTex; //the depth texture

            struct v2f
            {
                float4 pos : SV_POSITION;
                float4 projPos : TEXCOORD1; //Screen position of pos
                float2 uv : TEXCOORD0;

            };

            float4 _MainTex_ST;

            v2f vert(appdata_base v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.projPos = ComputeScreenPos(o.pos);
                o.uv = TRANSFORM_TEX (v.texcoord, _MainTex);
               
                return o;
            }
 
            half4 frag(v2f i) : COLOR
            {
                //Grab the depth value from the depth texture
                //Linear01Depth restricts this value to [0, 1]
                float depth = Linear01Depth (tex2Dproj(_CameraDepthTexture,
                                                             UNITY_PROJ_COORD(i.projPos)).r);
				//depth = 0;
   	            fixed4 tex = tex2D (_MainTex, i.uv);
                //if (tex.a > 0.8) discard;

                half4 c;
                c.r = (1-depth);
                c.g = (1-depth);
                c.b = (1-depth);
                c.a = 0; //1tex.a
                return c;
            }
 
            ENDCG
        }

    }

    FallBack "VertexLit"
    //FallBack "Diffuse"
}
