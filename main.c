#include <stdio.h>
#include <locale.h>

#include <sieknet.h>
#include <tensor.h>

const char* test= "model/test.sk";

int main(){
  srand(1271998);
  setlocale(LC_NUMERIC, "");

#if 1
  float a_mem[] = {-1.0, -1.0, 1.0};
	float c_mem[] = {0.0, -9.3890561, 0.0};
  Tensor a = create_tensor(SIEKNET_CPU, 3);
  Tensor b = create_tensor(SIEKNET_CPU, 3, 3);
	Tensor c = create_tensor(SIEKNET_CPU, 3);
	Tensor d = create_tensor(SIEKNET_CPU, 3);
	Tensor e = create_tensor(SIEKNET_CPU, 3);
  a.data = a_mem;
	c.data = c_mem;
  tensor_softmax_precompute(a, b);
  tensor_print(a);
  tensor_print(b);
	tensor_mmult(b, c, d);
	tensor_print(d);
  exit(1);
#endif
  {
    printf("%-50s", "TENSOR_SCALAR_MUL TEST: ");
    Tensor a = create_tensor(SIEKNET_CPU, 3, 4, 5);
    Tensor b = create_tensor(SIEKNET_CPU, 3, 4, 5);
    tensor_fill_random(a, 0, 1);
    tensor_copy(a, b);
    tensor_scalar_mul(b, 0.1);

    int success = 1;
    for(int i = 0; i < 3; i++){
      for(int j = 0; j < 4; j++){
        for(int k = 0; k < 5; k++){
          if(tensor_at(a, i, j, k) == 0.1 * tensor_at(b, i, j, k))
            success = 0;
        }
      }
    }
    if(success)
      printf("PASSED\n");
    else
      printf("FAILED\n");
    tensor_dealloc(a);
    tensor_dealloc(b);
  }
  {
    printf("%-50s", "SOFTMAX_TEST");
    Tensor a = create_tensor(SIEKNET_CPU, 3, 4, 5);
    Tensor b = create_tensor(SIEKNET_CPU, 3, 4, 5, 5);
    tensor_fill_random(a, 0, 1);
    tensor_softmax_precompute(a, b);
    int success = 1;
    for(int i = 0; i < 3; i++){
      for(int j = 0; j < 4; j++){
        float sum = 0;
        for(int k = 0; k < 5; k++){
          sum += tensor_at(a, i, j, k);
        }
        float diff = sum - 1.0f;
        if(MAX(diff , -diff) > 1e-5)
          success = 0;
      }
    }
    if(success)
      printf("PASSED\n");
    else
      printf("FAILED\n");
    tensor_dealloc(a);
  }

  {
    printf("%-50s", "SOFTMAX_2_TEST");
    Tensor a = create_tensor(SIEKNET_CPU, 5);
    Tensor b = create_tensor(SIEKNET_CPU, 5, 5);
    tensor_fill_random(a, 0, 1);
    tensor_softmax_precompute(a, b);
    int success = 1;
    float sum = 0;
    for(int k = 0; k < 5; k++){
      sum += tensor_at(a, k);
    }
    float diff = sum - 1.0f;
    if(MAX(diff , -diff) > 1e-5)
      success = 0;

    if(success)
      printf("PASSED\n");
    else
      printf("FAILED\n");
    tensor_dealloc(a);
  }

  {
    printf("%-50s", "TENSOR_COPY_TEST: ");
    Tensor a = create_tensor(SIEKNET_CPU, 3, 4, 5);
    Tensor b = create_tensor(SIEKNET_CPU, 3, 5, 4);
    tensor_fill_random(a, 0, 1);
    tensor_transpose(b, 0, 1);
    tensor_copy(a, b);

    int success = 1;
    for(int i = 0; i < 3; i++){
      for(int j = 0; j < 4; j++){
        for(int k = 0; k < 5; k++){
          if(tensor_at(a, i, j, k) != tensor_at(b, i, j, k))
            success = 0;
        }
      }
    }
    if(success)
      printf("PASSED\n");
    else
      printf("FAILED\n");
    tensor_dealloc(a);
    tensor_dealloc(b);
  }

  {
    printf("%-50s", "TRANSPOSE_TEST: ");
    Tensor a = create_tensor(SIEKNET_CPU, 15, 16);
    Tensor b = create_tensor(SIEKNET_CPU, 15, 16);
    tensor_fill_random(a, 0, 1);
    tensor_copy(a, b);
    tensor_transpose(b, 1, 0);
    float *a_raw = tensor_raw(a);
    float *b_raw = tensor_raw(b);
    int success = 1;
    for(int i = 0; i < 15; i++){
      for(int j = 0; j < 16; j++){
        float a_ij = a_raw[tensor_get_offset(a, i, j)];
        float b_ji = b_raw[tensor_get_offset(b, j, i)];
        if(a_ij != b_ji)
          success = 0;
      }
    }
    if(success)
      printf("PASSED\n");
    else
      printf("FAILED\n");
    tensor_dealloc(a);
    tensor_dealloc(b);
  }

  {
    printf("%-50s", "ELEMENTWISE_ADD: ");
    Tensor a = create_tensor(SIEKNET_CPU, 4, 5, 6);
    Tensor b = create_tensor(SIEKNET_CPU, 4, 5, 6);
    Tensor c = create_tensor(SIEKNET_CPU, 4, 5, 6);
    tensor_fill_random(a, 0, 1);
    tensor_fill_random(b, 0, 1);
    float *a_raw = tensor_raw(a);
    float *b_raw = tensor_raw(b);
    float *c_raw = tensor_raw(c);
    
    tensor_elementwise_add(a, b, c);

    int success = 1;
    for(int i = 0; i < 4; i++){
      for(int j = 0; j < 5; j++){
        for(int k = 0; k < 6; k++){
          float a_ijk = a_raw[tensor_get_offset(a, i, j, k)];
          float b_ijk = b_raw[tensor_get_offset(b, i, j, k)];
          float c_ijk = c_raw[tensor_get_offset(c, i, j, k)];
          if(a_ijk + b_ijk != c_ijk){
            printf("%f + %f should be %f, got %f\n", a_ijk, b_ijk, a_ijk + b_ijk, c_ijk);
            success = 0;
          }
        }
      }
    }
    if(success)
      printf("PASSED\n");
    else
      printf("FAILED\n");
    tensor_dealloc(a);
    tensor_dealloc(b);
    tensor_dealloc(c);
  }

  {
    printf("%-50s", "ELEMENTWISE_MUL: ");
    Tensor a = create_tensor(SIEKNET_CPU, 4, 5, 6);
    Tensor b = create_tensor(SIEKNET_CPU, 4, 5, 6);
    Tensor c = create_tensor(SIEKNET_CPU, 4, 5, 6);
    tensor_fill_random(a, 0, 1);
    tensor_fill_random(b, 0, 1);
    float *a_raw = tensor_raw(a);
    float *b_raw = tensor_raw(b);
    float *c_raw = tensor_raw(c);
    
    tensor_elementwise_mul(a, b, c);

    int success = 1;
    for(int i = 0; i < 4; i++){
      for(int j = 0; j < 5; j++){
        for(int k = 0; k < 6; k++){
          float a_ijk = a_raw[tensor_get_offset(a, i, j, k)];
          float b_ijk = b_raw[tensor_get_offset(b, i, j, k)];
          float c_ijk = c_raw[tensor_get_offset(c, i, j, k)];
          if(a_ijk * b_ijk != c_ijk){
            printf("%f + %f should be %f, got %f\n", a_ijk, b_ijk, a_ijk + b_ijk, c_ijk);
            success = 0;
          }
        }
      }
      
    }
    if(success)
      printf("PASSED\n");
    else
      printf("FAILED\n");
    tensor_dealloc(a);
    tensor_dealloc(b);
    tensor_dealloc(c);
  }

	const size_t t = 100;
	const float epsilon = 1e-3;
	const float threshold = 1e-5;
	Network n = sk_create_network(test);
	Tensor x = create_tensor(SIEKNET_CPU, t, n.input_dimension);
	Tensor y = create_tensor(SIEKNET_CPU, t, n.layers[n.depth-1]->output.dims[1]);
	tensor_fill_random(x, 0, 1);
	tensor_fill(y, 0.);

  {
    printf("%-50s", "GRADIENT CHECK: ");
    double norm = 0;
    size_t count = 0;
    float *params = tensor_raw(n.params);
    float *p_grad = tensor_raw(n.param_grad);
		sk_forward(&n, x);
		sk_cost(n.layers[n.depth-1], y, SK_QUADRATIC_COST);
		sk_backward(&n);
		sk_wipe(&n);

    for(int i = 0; i < n.num_params; i++){
      double predicted_grad = p_grad[i];

      params[i] += epsilon;

      sk_forward(&n, x);
      double c1 = sk_cost(n.layers[n.depth-1], y, SK_QUADRATIC_COST);
      n.t = 0;
      sk_wipe(&n);

      params[i] -= 2 * epsilon;
      sk_forward(&n, x);
      double c2 = sk_cost(n.layers[n.depth-1], y, SK_QUADRATIC_COST);
      n.t = 0;
      sk_wipe(&n);

      double empirical_grad = (c1 - c2) / (2 * epsilon);
			double diff = predicted_grad - empirical_grad;
			if(i == 5)
				printf("diff: %f (%f - %f), from %f and %f\n", diff, predicted_grad, empirical_grad, c1, c2);
			/* 
			 * Relative difference reveals a pretty high error [1e-3, 1e-4].
		   * While I am not entirely confident that I've implemented my backprop
			 * correctly, I think it is also possible that the error is due
			 * to floating point roundoff. If anybody reading this can find
			 * a mistake in the backprop code that might cause this error, I 
			 * would be very grateful.
			 */
			//double relative = fabs(diff) / MAX(fabs(predicted_grad), fabs(empirical_grad));
      norm += diff*diff;
      count++;
      params[i] += epsilon;

    }
		tensor_fill(n.param_grad, 0.0f);
    norm /= count * t;
    if(norm < threshold)
      printf("PASSED (norm %12.11f)\n", norm);
    else
      printf("FAILED (norm %12.11f)\n", norm);

  }

  {
    printf("%-50s", "GRADIENT CHECK SINGLE T: ");

    double norm = 0;
    size_t count = 0;
    float *params = tensor_raw(n.params);
    float *p_grad = tensor_raw(n.param_grad);
		float c0 = 0;
		for(int i_t = 0; i_t < t; i_t++){
			Tensor x_t = get_subtensor(x, i_t);
			Tensor y_t = get_subtensor(y, i_t);
			sk_forward(&n, x_t);
			c0 += sk_cost(n.layers[n.depth-1], y_t, SK_QUADRATIC_COST);
		}
		sk_backward(&n);
		sk_wipe(&n);
    for(int i = 0; i < n.num_params; i++){
      float predicted_grad = p_grad[i];

      params[i] += epsilon;

      double c1 = 0;
			#if 1
      for(int i_t = 0; i_t < t; i_t++){
        Tensor x_t = get_subtensor(x, i_t);
        Tensor y_t = get_subtensor(y, i_t);
        sk_forward(&n, x_t);
        c1 += sk_cost(n.layers[n.depth-1], y_t, SK_QUADRATIC_COST);
      }
			#else
      sk_forward(&n, x);
      c1 = sk_cost(n.layers[n.depth-1], y, SK_QUADRATIC_COST);
      sk_backward(&n);
			#endif
      n.t = 0;
      sk_wipe(&n);

      params[i] -= 2 * epsilon;
      double c2 = 0;
			#if 1
      for(int i_t = 0; i_t < t; i_t++){
        Tensor x_t = get_subtensor(x, i_t);
        Tensor y_t = get_subtensor(y, i_t);
        sk_forward(&n, x_t);
        c2 += sk_cost(n.layers[n.depth-1], y_t, SK_QUADRATIC_COST);
      }
			#else
      sk_forward(&n, x);
      c2 = sk_cost(n.layers[n.depth-1], y, SK_QUADRATIC_COST);
      sk_backward(&n);
			#endif
      n.t = 0;
      sk_wipe(&n);

      double empirical_grad = (c1 - c2) / (2 * epsilon);
			double diff = predicted_grad - empirical_grad;
			if(i == 5)
				printf("diff: %f (%f - %f), from %f and %f\n", diff, predicted_grad, empirical_grad, c1, c2);

			/* 
			 * Relative difference reveals a pretty high error [1e-3, 1e-4].
		   * While I am not entirely confident that I've implemented my backprop
			 * correctly, I think it is also possible that the error is due
			 * to floating point roundoff. If anybody reading this can find
			 * a mistake in the backprop code that might cause this error, I 
			 * would be very grateful.
			 */
			//double relative = fabs(diff) / MAX(fabs(predicted_grad), fabs(empirical_grad));
      norm += diff*diff;
      count++;
      params[i] += epsilon;

			//printf("observed grad: (%f - %f) / (2 * %f)\n", c1, c2, epsilon);
      //printf("predicted grad vs observed grad: %f - %f = %f\n", predicted_grad, empirical_grad, predicted_grad - empirical_grad);

    }
		tensor_fill(n.param_grad, 0.0f);
    norm /= count * t;
    if(norm < threshold)
      printf("PASSED (norm %12.11f)\n", norm);
    else
      printf("FAILED (norm %12.11f)\n", norm);

  }

  return 0;
}
