#include <stdio.h>
#include <locale.h>

#include <sieknet.h>
#include <tensor.h>

const char* test= "model/test.sk";

int main(){
  srand(0);
  setlocale(LC_NUMERIC, "");

  {
    printf("TENSOR_COPY_TEST: ");
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
    printf("TRANSPOSE_TEST: ");
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
    printf("ELEMENTWISE_ADD: ");
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
    printf("ELEMENTWISE_MUL: ");
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

  {
    printf("GRADIENT CHECK: ");
    size_t t = 50;
    Network n = sk_create_network(test);
    Tensor x = create_tensor(SIEKNET_CPU, t, n.input_dimension);
    Tensor y = create_tensor(SIEKNET_CPU, t, n.layers[n.depth-1]->output.dims[1]);
		tensor_fill_random(x, 0, 0.3);
		tensor_fill_random(y, 0.5, 0.1);

    float norm = 0;
    size_t count = 0;
    float *params = tensor_raw(n.params);
    float *p_grad = tensor_raw(n.param_grad);
    float epsilon = 1e-3;
    for(int i = 0; i < n.num_params; i++){
      sk_forward(&n, x);
      sk_cost(n.layers[n.depth-1], y, SK_QUADRATIC_COST);
      sk_backward(&n);

      sk_wipe(&n);

      float predicted_grad = p_grad[i];

      params[i] += epsilon;

      sk_forward(&n, x);
      float c1 = sk_cost(n.layers[n.depth-1], y, SK_QUADRATIC_COST);
      n.t = 0;
      sk_wipe(&n);

      params[i] -= 2 * epsilon;
      sk_forward(&n, x);
      float c2 = sk_cost(n.layers[n.depth-1], y, SK_QUADRATIC_COST);
      n.t = 0;
      sk_wipe(&n);

      float empirical_grad = (c1 - c2) / (2 * epsilon);
      norm += (predicted_grad - empirical_grad) * (predicted_grad - empirical_grad);
      count++;
      params[i] += epsilon;

			tensor_fill(n.param_grad, 0.0f);
    }
    norm /= count;
    if(norm < 1e-3)
      printf("PASSED (norm %12.11f)\n", norm);
    else
      printf("FAILED (norm %5.4f)\n", norm);

  }

  {
    printf("GRADIENT CHECK SINGLE T: ");
    size_t t = 50;
    Network n = sk_create_network(test);
    Tensor x = create_tensor(SIEKNET_CPU, t, n.input_dimension);
    Tensor y = create_tensor(SIEKNET_CPU, t, n.layers[n.depth-1]->output.dims[1]);
		tensor_fill_random(x, 0, 0.3);
		tensor_fill_random(y, 0.5, 0.1);

    float norm = 0;
    size_t count = 0;
    float *params = tensor_raw(n.params);
    float *p_grad = tensor_raw(n.param_grad);
    float epsilon = 1e-3;
    for(int i = 0; i < n.num_params; i++){
      for(int i_t = 0; i_t < t; i_t++){
        Tensor x_t = get_subtensor(x, i_t);
        Tensor y_t = get_subtensor(y, i_t);
        sk_forward(&n, x_t);
        sk_cost(n.layers[n.depth-1], y_t, SK_QUADRATIC_COST);
      }
      sk_backward(&n);

      sk_wipe(&n);

      float predicted_grad = p_grad[i];

      params[i] += epsilon;

      float c1 = 0;
      for(int i_t = 0; i_t < t; i_t++){
        Tensor x_t = get_subtensor(x, i_t);
        Tensor y_t = get_subtensor(y, i_t);
        sk_forward(&n, x_t);
        c1 += sk_cost(n.layers[n.depth-1], y_t, SK_QUADRATIC_COST);
      }
      n.t = 0;
      sk_wipe(&n);

      params[i] -= 2 * epsilon;
      float c2 = 0;
      for(int i_t = 0; i_t < t; i_t++){
        Tensor x_t = get_subtensor(x, i_t);
        Tensor y_t = get_subtensor(y, i_t);
        sk_forward(&n, x_t);
        c2 += sk_cost(n.layers[n.depth-1], y_t, SK_QUADRATIC_COST);
      }
      n.t = 0;
      sk_wipe(&n);

      float empirical_grad = (c1 - c2) / (2 * epsilon);
      norm += (predicted_grad - empirical_grad) * (predicted_grad - empirical_grad);
      count++;
      params[i] += epsilon;

      //printf("predicted grad vs observed grad: %f - %f = %f\n", predicted_grad, empirical_grad, predicted_grad - empirical_grad);

			tensor_fill(n.param_grad, 0.0f);
    }
    norm /= count;
    if(norm < 1e-3)
      printf("PASSED (norm %12.11f)\n", norm);
    else
      printf("FAILED (norm %5.4f)\n", norm);

  }

  return 0;
}
