SRC = main.cu
EXEC = raytracer

all : ${EXEC}

${EXEC} : ${SRC}
	nvcc ${SRC} -o ${EXEC}

tests : ${EXEC}
	${EXEC}
	"C:\Program Files\ImageGlass\ImageGlass.exe" result.ppm

git : 
	git add *.h *.cuh *.cu Makefile *.md *.obj
	git status
	git commit -m "updated"
	git push