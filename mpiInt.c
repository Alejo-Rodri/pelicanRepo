#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* Función a integrar */
double f(double x) {
    return (0.2 * (x - 4) * (x - 4) * (x - 4)) - (2 * x) + 12;
}

/* Regla trapezoidal */
double Trap(double local_a, double local_b, int local_n, double h) {
    double estimate = (f(local_a) + f(local_b)) / 2.0;
    for (int i = 1; i < local_n; i++) {
        double x = local_a + i * h;
        estimate += f(x);
    }
    estimate = estimate * h;
    return estimate;
}

int main(int argc, char** argv) {
    int my_rank, comm_sz, local_n;
    double a = 0.0, b = 10.0;  // Límites de la integral
    double n, h, local_a, local_b;
    double local_int, total_int;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // El proceso 0 obtiene el número de trapecios de la entrada
    if (my_rank == 0) {
        if (argc != 2) {
            printf("Uso: mpirun -np <num_procesos> %s <numero de trapecios>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);  // Finalizar si no hay suficientes argumentos
        }
        n = atof(argv[1]);  // Lee el número de trapecios
    }

    // Difundir el valor de n a todos los procesos
    MPI_Bcast(&n, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calcular el tamaño de cada paso y el número de trapecios por proceso
    h = (b - a) / n;
    local_n = n / comm_sz;

    // Cada proceso calcula sus propios límites de integración
    local_a = a + my_rank * local_n * h;
    local_b = local_a + local_n * h;

    // Cada proceso calcula su parte de la integral
    local_int = Trap(local_a, local_b, local_n, h);

    // Reducir todos los resultados locales en el proceso 0
    MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // El proceso 0 imprime el resultado
    if (my_rank == 0) {
        printf("Con n = %.0f trapezoides, la integral entre %.0f y %.0f es aproximadamente: %f\n", n, a, b, total_int);
    }

    MPI_Finalize();
    return 0;
}
