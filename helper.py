from random import randint

# ฟังก์ชั่นสุ่มสี
def getRandomColors(n):
    colors = []
    for i in range(n):
        colors.append((randint(0,n-1)*(255//n),randint(0,n-1)*(255//n),randint(0,n-1)*(255//n)))
    return colors

if __name__ == "__main__":
    print(getRandomColors(20))