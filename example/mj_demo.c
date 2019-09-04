#include <env.h>

int main(){
  Environment e = create_walker2d_env("cgym/assets/walker2d.xml");

  e.reset(e);
  e.seed(e);

  while(1){
    float a[e.action_space];
    for(int i = 0; i < e.action_space; i++)
      a[i] = 0;
    e.step(e, a);
    e.render(e);
    if(*e.done){
      e.reset(e);
      e.seed(e);
    }
  }

}
