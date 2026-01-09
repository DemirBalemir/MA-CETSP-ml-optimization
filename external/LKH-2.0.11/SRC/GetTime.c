#include <time.h>

double GetTime(void)
{
    return (double)clock() / CLOCKS_PER_SEC;
}
