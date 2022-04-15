// Summary: (FA) && (a != b)
int lib(int a, int b) {
    int result = 0;
    if (FA) {
        result += b; // result += a;
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

int client(int a, int b) {
    // Summary:
    // (FA && !FB) && (a + 1 != b)
    if (FA) {
        if (FB) {
            return lib(b, b);
        }
        return lib(a + 1, b);
    } else {
        return lib(a, b);
    }
}
