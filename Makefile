SRC = main.cu
EXEC = raytracer

all : ${EXEC}

${EXEC} : ${SRC}
	nvcc ${SRC} -o ${EXEC}

tests : ${EXEC}
	${EXEC}