#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  float indices[N];
  for (int j = 0; j < N; j++) {
    indices[j] = (float)j; // 0.0, 1.0, 2.0, ... 15.0 という連番を作る
  }

  __m512 x_vec = _mm512_load_ps(x);
  __m512 y_vec = _mm512_load_ps(y);
  __m512 m_vec = _mm512_load_ps(m);
  __m512 j_vec = _mm512_load_ps(indices);
  // __m512 fx_vec = _mm512_load_ps(fx);
  // __m512 fy_vec = _mm512_load_ps(fy);

  for(int i=0; i<N; i++) {

    __m512 i_vec = _mm512_set1_ps((float)i); //iを全要素にセットしたベクトル
    __mmask16 mask = _mm512_cmp_ps_mask(i_vec, j_vec, _MM_CMPINT_NE); //iとjが等しくない要素だけを選ぶマスク

    __m512 rx = _mm512_sub_ps(_mm512_set1_ps(x[i]), x_vec); //rxの一括計算
    __m512 ry = _mm512_sub_ps(_mm512_set1_ps(y[i]), y_vec); //ryの一括計算

    //__m512 r = _mm512_rsqrt14_ps(_mm512_add_ps(_mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry))); //rの一括計算
    __m512 r2 = _mm512_add_ps(_mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry));
    __m512 r = _mm512_rsqrt14_ps(r2);
    __m512 r3 = _mm512_mul_ps(r, _mm512_mul_ps(r, r));

    __m512 fx_temp = _mm512_mul_ps(rx, _mm512_mul_ps(m_vec, r3));
    __m512 fy_temp = _mm512_mul_ps(ry, _mm512_mul_ps(m_vec, r3));

    // ★ここでブレンド！ 条件を満たさない（i==j）場所のNaNを「0.0」にすり替える
    __m512 zero = _mm512_setzero_ps();
    fx_temp = _mm512_mask_blend_ps(mask, zero, fx_temp);
    fy_temp = _mm512_mask_blend_ps(mask, zero, fy_temp);

    fx[i] -= _mm512_reduce_add_ps(fx_temp);
    fy[i] -= _mm512_reduce_add_ps(fy_temp);
    
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
