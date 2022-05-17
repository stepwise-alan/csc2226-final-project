int lib_new(int x, int y) {
    int result = 0;
    if (A) {
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

int lib_old(int x, int y) {
    int result = 0;
    if (A) {
        result += x;
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

void client(int x, int y) {
    if (A) {
        if (B) {
            assert(lib_new(y, y) == lib_old(y, y));
        }
        assert(lib_new(x + 1, y) == lib_old(x + 1, y));
    } else {
        assert(lib_new(x, y) == lib_old(x, y));
    }
}

int main() {
    int x;
    int y;
    client(x, y);
}