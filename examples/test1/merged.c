int lib_new(int a, int b) {
    int result = 0;
    if (FA) {
        result += b;
    } else {
        result += b;
    }
    if (FB) {
        result += a;
    } else {
        result += b;
    }
    if (FC) {
        result += a;
    } else {
        result += b;
    }
    return result;
}

int lib_old(int a, int b) {
    int result = 0;
    if (FA) {
        result += a;
    } else {
        result += b;
    }
    if (FB) {
        result += a;
    } else {
        result += b;
    }
    if (FC) {
        result += a;
    } else {
        result += b;
    }
    return result;
}

void client(int a, int b) {
    if (FA) {
        if (FB) {
            assert(lib_new(b, b) == lib_old(b, b));
        }
        assert(lib_new(a + 1, b) == lib_old(a + 1, b));
    } else {
        assert(lib_new(a, b) == lib_old(a, b));
    }
}

int main() {
    int a;
    int b;
    client(a, b);
}