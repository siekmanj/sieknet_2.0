#include <getopt.h>
#include <stdio.h>
#include <string.h>    /* for strcmp   */
#include <stdint.h>    /* for uint32_t */
#include <arpa/inet.h> /* for htonl    */
#include <time.h>

#include <sieknet.h>   /* for the fun stuff */
#include <optimizer.h>

size_t clock_us(){
  struct timespec start;
  clock_gettime(CLOCK_REALTIME, &start);
  return start.tv_sec * 1e6 + start.tv_nsec / 1e3;
}

/*
 * Loads mnist image data from a file into a tensor.
 */
Tensor binary_to_image_tensor(const char *fname){
  FILE *data = fopen(fname, "rb");

  if(!data) SK_ERROR("Unable to open '%s'\n", fname);

  uint32_t magic_num;
  if(!fread(&magic_num, sizeof(uint32_t), 1, data)) SK_ERROR("Unable to read from '%s'.\n", fname);
  if(htonl(magic_num) != 0x803) SK_ERROR("Invalid mnist data file - expected first four bytes to be 0x803, but got 0x%x", htonl(magic_num));

  uint32_t img_dims[2] = {0};
  uint32_t num_imgs = 0;
  if(!fread(&num_imgs, sizeof(uint32_t), 1, data)) SK_ERROR("Unable to read from '%s'.\n", fname);
  if(!fread(&img_dims, sizeof(uint32_t), 2, data)) SK_ERROR("Unable to read from '%s'.\n", fname);

  num_imgs = htonl(num_imgs);
  img_dims[0] = htonl(img_dims[0]);
  img_dims[1] = htonl(img_dims[1]);

  Tensor ret = create_tensor(SIEKNET_CPU, num_imgs, img_dims[0] * img_dims[1]);

  float *raw = tensor_raw(ret);
  for(int i = 0; i < num_imgs * img_dims[0] * img_dims[1]; i++){
    uint8_t num;
    if(!fread(&num, sizeof(uint8_t), 1, data)) SK_ERROR("Failed to read byte from '%s' on byte %d.\n", fname, i+16);
    raw[i] = num / 255.0f;
  }

  return ret;
}

/*
 * Load mnist label data into a tensor.
 */
Tensor binary_to_label_tensor(const char *fname){
  FILE *data = fopen(fname, "rb");

  if(!data) SK_ERROR("Unable to open '%s'\n", fname);

  uint32_t magic_num;
  if(!fread(&magic_num, sizeof(uint32_t), 1, data)) SK_ERROR("Unable to read from '%s'.\n", fname);
  if(htonl(magic_num) != 0x801) SK_ERROR("Invalid mnist label file - expected first four bytes to be 0x801, but got 0x%x", htonl(magic_num));

  uint32_t num_labels = 0;
  if(!fread(&num_labels, sizeof(uint32_t), 1, data)) SK_ERROR("Unable to read from '%s'.\n", fname);

  num_labels = htonl(num_labels);

  Tensor ret = create_tensor(SIEKNET_CPU, num_labels, 10);

  for(int i = 0; i < num_labels; i++){
    uint8_t num;
    if(!fread(&num, sizeof(uint8_t), 1, data)) SK_ERROR("Failed to read byte from '%s' on byte %d.\n", fname, i+8);

    if(num > 9)
      SK_ERROR("Not really sure how this happened, but got a label for image %d that was %d.", i, num);

    *tensor_raw(get_subtensor(ret, i, num)) = 1.0f;
  }

  return ret;
}

int main(int argc, char **argv){
  char *mnist_data        = NULL;
  char *mnist_data_labels = NULL;
  char *mnist_test        = NULL;
  char *mnist_test_labels = NULL;

  setbuf(stdout, NULL);

  /*
   * Read command line options to get location of mnist data
   */
  int args_read = 0;
  while(1){
    static struct option long_options[] = {
      {"training_set",    required_argument, 0,  0 },
      {"training_labels", required_argument, 0,  0 },
      {"test_set",        required_argument, 0,  0 },
      {"test_labels",     required_argument, 0,  0 },
      {0,                 0,                 0,  0 },
    };

    int opt_idx;
    char c = getopt_long(argc, argv, "", long_options, &opt_idx);
    if(!c){
      if(!strcmp(long_options[opt_idx].name, "training_set"))     mnist_data        = optarg;
      if(!strcmp(long_options[opt_idx].name, "training_labels"))  mnist_data_labels = optarg;
      if(!strcmp(long_options[opt_idx].name, "test_set"))         mnist_test        = optarg;
      if(!strcmp(long_options[opt_idx].name, "test_labels"))      mnist_test_labels = optarg;
      args_read++;
    }else if(c == -1) break;
  }

  if(args_read < 4){
    if(!mnist_data)
      printf("Missing arg: --training_set [binary file]\n");
    if(!mnist_data_labels)
      printf("Missing arg: --training_labels [binary file]\n");
    if(!mnist_test)
      printf("Missing arg: --test_set [binary file]\n");
    if(!mnist_test_labels)
      printf("Missing arg: --test_set_labels [binary file]\n");
    exit(1);
  }

  /*
   * Open and extract the binary info from the files
   */
  Tensor data = binary_to_image_tensor(mnist_data);
  Tensor data_labels = binary_to_label_tensor(mnist_data_labels);

  if(data.dims[0] != data_labels.dims[0])
    SK_ERROR("data image tensor and label tensor do not match in length - %lu vs %lu\n", data.dims[0], data_labels.dims[0]);

  Tensor test = binary_to_image_tensor(mnist_test);
  Tensor test_labels = binary_to_label_tensor(mnist_test_labels);

  if(test.dims[0] != test_labels.dims[0])
    SK_ERROR("test image tensor and label tensor do not match in length - %lu vs %lu\n", test.dims[0], test_labels.dims[0]);


  const size_t dataset_len = data.dims[0];
  const size_t testset_len = test.dims[0];
  const size_t batch_size  = 16;
  const size_t epochs      = 3;
  const size_t batches     = dataset_len / batch_size;

  Network n = sk_create_network("model/mnist.sk");
  Optimizer o = create_optimizer(n.params, n.param_grad, SK_SGD);
  o.lr = 0.01;

  printf("   _____ ____________ __ _   ______________\n");
  printf("  / ___//  _/ ____/ //_// | / / ____/_	__/\n");
  printf("  \\__ \\ / // __/ / ,<  /  |/ / __/   / /   \n");
  printf(" ___/ // // /___/ /| |/ /|  / /___  / /    \n");
  printf("/____/___/_____/_/ |_/_/ |_/_____/ /_/	   \n");
  printf("																					 \n");
  printf("MNIST demo - trains a classifier on the MNIST dataset.\n");

  Layer *output_layer = sk_layer_from_name(&n, "softmax_layer");

  float avg_batch_time = 0;
  for(int epoch = 0; epoch < epochs; epoch++){
    printf("Commencing epoch %d:\n", epoch);

    float epoch_cost = 0.0f;
    for(int batch = 0; batch < batches; batch++){
      size_t start = clock_us();
      for(int i = 0; i < batch_size; i++){
        int rand_idx = rand() % dataset_len;
        Tensor x = get_subtensor(data, rand_idx);
        Tensor y = get_subtensor(data_labels, rand_idx);

        sk_forward(&n, x);
        if(output_layer)
          epoch_cost += sk_cost(output_layer, y, SK_CROSS_ENTROPY_COST);
      }
      sk_backward(&n);
      o.step(o);

      float elapsed = (clock_us()- start)/1e6;
      avg_batch_time = (avg_batch_time * (epoch * batches + batch) + elapsed)/(epoch * batches + batch + 1);
      float completion = ((float)epoch * batches + batch) / (epochs * batches);
      float time_left = (1 - completion) * epochs * batches * avg_batch_time;
      int min_left = ((int)time_left)/(60);
      int sec_left = ((int)time_left - min_left*60);
      
      printf("Did batch %'6d of %'6lu (%'3dm %'2ds remain)\t\r", batch, batches, min_left, sec_left);
    }
    printf("Epoch %d complete, cost %f \n", epoch, epoch_cost / dataset_len);
  }
  size_t correct = 0;
  for(int i = 0; i < testset_len; i++){
    Tensor x = get_subtensor(test, i);
    Tensor y = get_subtensor(test_labels, i);
    n.t = 0;
    sk_forward(&n, x);
    int guess = tensor_argmax(get_subtensor(output_layer->output, 0));
    int label = tensor_argmax(y);
    if(guess == label)
      correct++;
  }
  printf("%6lu of %6lu correct (%3.2f%%)\n", correct, testset_len, 100 * (double)correct / testset_len);
}
