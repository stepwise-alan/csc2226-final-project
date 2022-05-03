int main()
{
  int x;
  int y;
  lib(x, y);
}

int lib(int x, int y)
{
  int CLEVER_ret_2_old = 0;
  int CLEVER_ret_2_new = 0;
  int CLEVER_ret_1_old = 0;
  int CLEVER_ret_1_new = 0;
  int CLEVER_ret_0_old = 0;
  int CLEVER_ret_0_new = 0;
  if (A)
  {
    if (B)
    {
      CLEVER_ret_0_old = lib_old(y, y);
      CLEVER_ret_0_new = lib_new(y, y);
    }
    else
    {
      CLEVER_ret_1_old = lib_old(x + 1, y);
      CLEVER_ret_1_new = lib_new(x + 1, y);
    }
  }
  else
  {
    CLEVER_ret_2_old = lib_old(x, y);
    CLEVER_ret_2_new = lib_new(x, y);
  }
  assert(CLEVER_ret_0_old == CLEVER_ret_0_new);
  assert(CLEVER_ret_1_old == CLEVER_ret_1_new);
  assert(CLEVER_ret_2_old == CLEVER_ret_2_new);
}

int lib_old(int x, int y)
{
  int result = 0;
  if (A)
  {
    result += y;
  }
  else
  {
    result += y;
  }
  if (B)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (C)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (D)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (E)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (F)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (G)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (H)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (I)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (J)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (K)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (L)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (M)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (N)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (O)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (P)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (Q)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (R)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  return result;
}

int lib_new(int x, int y)
{
  int result = 0;
  if (A)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (B)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (C)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (D)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (E)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (F)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (G)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (H)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (I)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (J)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (K)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (L)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (M)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (N)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (O)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (P)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (Q)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  if (R)
  {
    result += x;
  }
  else
  {
    result += y;
  }
  return result;
}
