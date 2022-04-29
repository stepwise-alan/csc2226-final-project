// FA: (x != y)
int lib(int x, int y) {
    int result = 0;
    if (FA) {
//      result += x;
        result += y;
    } else {
        result += y;
    }
    if (FB) {
        result += x;
    } else {
        result += y;
    }
    if (FC) {
        result += x;
    } else {
        result += y;
    }
    return result;
}

int client(int x, int y) {
    // FA and !FB: x + 1 != y
    if (FA) {
        // FA and !FB: x + 1 != y
        if (FB) {
            return lib(y, y);
        } else {
            // !FA: x + 1 != y
            return lib(x + 1, y);
        }
    } else {
        // !FA: x != y
        return lib(x, y);
    }
}
