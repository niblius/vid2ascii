#define LINE_HEIGHT 16
#define SYMBOL_WIDTH 8
#define CHARACTER_COUNT 95

typedef struct Character {
  char codePoint;
  int x, y, width, height, originX, originY;
} Character;

Character characters_Arial[] = {
  {' ', 85, 58, 3, 3, 1, 1},
  {'!', 128, 36, 4, 11, 0, 10},
  {'"', 40, 58, 6, 5, 1, 10},
  {'#', 70, 25, 9, 11, 1, 10},
  {'$', 69, 0, 9, 12, 1, 10},
  {'%', 125, 0, 13, 11, 1, 10},
  {'&', 67, 14, 10, 11, 1, 10},
  {'\'', 46, 58, 4, 5, 1, 10},
  {'(', 11, 0, 6, 14, 1, 10},
  {')', 17, 0, 6, 14, 1, 10},
  {'*', 20, 58, 7, 7, 1, 10},
  {'+', 37, 47, 9, 9, 1, 9},
  {',', 36, 58, 4, 6, 0, 3},
  {'-', 59, 58, 6, 4, 1, 5},
  {'.', 71, 58, 4, 4, 0, 3},
  {'/', 104, 36, 6, 11, 1, 10},
  {'0', 79, 25, 9, 11, 1, 10},
  {'1', 88, 25, 9, 11, 1, 10},
  {'2', 97, 25, 9, 11, 1, 10},
  {'3', 106, 25, 9, 11, 1, 10},
  {'4', 115, 25, 9, 11, 1, 10},
  {'5', 124, 25, 9, 11, 1, 10},
  {'6', 133, 25, 9, 11, 1, 10},
  {'7', 0, 36, 9, 11, 1, 10},
  {'8', 9, 36, 9, 11, 1, 10},
  {'9', 18, 36, 9, 11, 1, 10},
  {':', 8, 58, 4, 9, 0, 8},
  {';', 132, 36, 4, 11, 0, 8},
  {'<', 46, 47, 9, 9, 1, 9},
  {'=', 27, 58, 9, 6, 1, 7},
  {'>', 55, 47, 9, 9, 1, 9},
  {'?', 27, 36, 9, 11, 1, 10},
  {'@', 55, 0, 14, 13, 1, 10},
  {'A', 77, 14, 10, 11, 1, 10},
  {'B', 87, 14, 10, 11, 1, 10},
  {'C', 12, 14, 11, 11, 1, 10},
  {'D', 23, 14, 11, 11, 1, 10},
  {'E', 97, 14, 10, 11, 1, 10},
  {'F', 36, 36, 9, 11, 1, 10},
  {'G', 34, 14, 11, 11, 1, 10},
  {'H', 107, 14, 10, 11, 1, 10},
  {'I', 136, 36, 4, 11, 0, 10},
  {'J', 72, 36, 8, 11, 1, 10},
  {'K', 117, 14, 10, 11, 1, 10},
  {'L', 45, 36, 9, 11, 1, 10},
  {'M', 0, 14, 12, 11, 1, 10},
  {'N', 127, 14, 10, 11, 1, 10},
  {'O', 45, 14, 11, 11, 1, 10},
  {'P', 0, 25, 10, 11, 1, 10},
  {'Q', 0, 0, 11, 14, 1, 10},
  {'R', 56, 14, 11, 11, 1, 10},
  {'S', 10, 25, 10, 11, 1, 10},
  {'T', 20, 25, 10, 11, 1, 10},
  {'U', 30, 25, 10, 11, 1, 10},
  {'V', 40, 25, 10, 11, 1, 10},
  {'W', 111, 0, 14, 11, 1, 10},
  {'X', 50, 25, 10, 11, 1, 10},
  {'Y', 60, 25, 10, 11, 1, 10},
  {'Z', 54, 36, 9, 11, 1, 10},
  {'[', 23, 0, 6, 14, 1, 10},
  {'\\', 110, 36, 6, 11, 1, 10},
  {']', 41, 0, 5, 14, 1, 10},
  {'^', 12, 58, 8, 7, 1, 10},
  {'_', 75, 58, 10, 3, 2, -1},
  {'`', 65, 58, 6, 4, 1, 11},
  {'a', 64, 47, 9, 9, 1, 8},
  {'b', 63, 36, 9, 11, 1, 10},
  {'c', 91, 47, 8, 9, 1, 8},
  {'d', 80, 36, 8, 11, 1, 10},
  {'e', 73, 47, 9, 9, 1, 8},
  {'f', 116, 36, 6, 11, 1, 10},
  {'g', 87, 0, 8, 12, 1, 8},
  {'h', 88, 36, 8, 11, 1, 10},
  {'i', 0, 47, 4, 11, 1, 10},
  {'j', 46, 0, 5, 14, 2, 10},
  {'k', 96, 36, 8, 11, 1, 10},
  {'l', 4, 47, 4, 11, 1, 10},
  {'m', 14, 47, 12, 9, 1, 8},
  {'n', 99, 47, 8, 9, 1, 8},
  {'o', 82, 47, 9, 9, 1, 8},
  {'p', 78, 0, 9, 12, 1, 8},
  {'q', 95, 0, 8, 12, 1, 8},
  {'r', 8, 47, 6, 10, 1, 9},
  {'s', 107, 47, 8, 9, 1, 8},
  {'t', 122, 36, 6, 11, 1, 10},
  {'u', 115, 47, 8, 9, 1, 8},
  {'v', 123, 47, 8, 9, 1, 8},
  {'w', 26, 47, 11, 9, 1, 8},
  {'x', 131, 47, 8, 9, 1, 8},
  {'y', 103, 0, 8, 12, 1, 8},
  {'z', 0, 58, 8, 9, 1, 8},
  {'{', 29, 0, 6, 14, 1, 10},
  {'|', 51, 0, 4, 14, 0, 10},
  {'}', 35, 0, 6, 14, 1, 10},
  {'~', 50, 58, 9, 4, 1, 6},
};
