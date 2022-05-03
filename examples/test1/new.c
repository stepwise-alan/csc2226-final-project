// A: (x != y)
int lib(int x, int y) {
    int result = 0;
    if (A) {
//      result += x;
        result += y;
    } else {
        result += y;
    }
    if (B) {
        result += x;
    } else {
        result += y;
    }
    if (C) {
        result += x;
    } else {
        result += y;
    }
    return result;
}

int client(int x, int y) {
    // A and !B: x + 1 != y
    if (A) {
        // A and !B: x + 1 != y
        if (B) {
            return lib(y, y);
        } else {
            // !A: x + 1 != y
            return lib(x + 1, y);
        }
    } else {
        // !A: x != y
        return lib(x, y);
    }
}
