#ifndef __VERFIY_HEADER__
#define __VERFIY_HEADER__

float getPrecision(float tmp)
{
    int tmpInt = (int)tmp;
    float eNum = 1.0e-6;
    if (abs(tmpInt) > 0)
    {
        while (tmpInt != 0)
        {
            tmpInt = (int)(tmpInt / 10);
            eNum *= 10;
        }
    }
    else
    {

        if (tmp == 0)
            return eNum;

        eNum = 1.0e-5;

        while (tmpInt == 0)
        {
            tmp *= 10;
            tmpInt = (int)(tmp);
            eNum /= 10;
        }
    }
    return eNum;
}

void conv2dcpu(float *pin, float *pwei, float *pout, int n, int c, int h, int w, int k, int r, int s, int u, int v, int p, int q)
{
    int oh = (h + 2 * p - r) / u + 1;
    int ow = (w + 2 * q - s) / v + 1;

    for (int nNum = 0; nNum < n; nNum++)
    {
        for (int kNum = 0; kNum < k; kNum++)
        {
            for (int i = 0; i < oh; i++)
            {
                for (int j = 0; j < ow; j++)
                {
                    double sum = 0.0;
                    int posh = i * u - p;
                    int posw = j * v - q;

                    for (int cNum = 0; cNum < c; cNum++)
                    {
                        for (int khNum = 0; khNum < r; khNum++)
                        {
                            for (int kwNum = 0; kwNum < s; kwNum++)
                            {
                                int posh_ori = posh + khNum;
                                int posw_ori = posw + kwNum;
                                if (posw_ori >= 0 && posh_ori >= 0 && posw_ori < w && posh_ori < h)
                                {
                                    sum += (double)(pin[nNum * c * h * w + cNum * (w * h) + posh_ori * w + posw_ori] * pwei[kNum * r * s * c + cNum * r * s + khNum * s + kwNum]);
                                }
                            }
                        }
                    }

                    pout[nNum * k * oh * ow + kNum * oh * ow + i * ow + j] = (float)sum;
                }
            }
        }
    }
}
#endif