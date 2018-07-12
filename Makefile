SRC = main.cu
EXEC = raytracer

all : ${EXEC}

${EXEC} : ${SRC}
	nvcc ${SRC} -o ${EXEC}

tests : ${EXEC}
	${EXEC}

git : 
	git add *.h *.cuh *.cu Makefile
	git commit -m "updated"
	git push