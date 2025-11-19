# © 2025 ETH Zürich, Jonas Heinzmann, Francesco Vicentini, Pietro Carrara, Laura De Lorenzis

from mpi4py import MPI
import sys

from dolfinx.io import VTXWriter
from dolfinx.fem import Function
from dolfinx.cpp.io import VTXMeshPolicy


def stdout(msg) -> None:
    """convenience function for printing (parallelized)"""
    if MPI.COMM_WORLD.rank == 0:
        print(msg)
        sys.stdout.flush()


class CSVWriter:
    """convenience class for writing CSV files (in parallel)"""

    def __init__(self, filename: str, header: str) -> None:
        self.filename = filename

        # open new file in write mode and write header
        if MPI.COMM_WORLD.rank == 0:
            with open(self.filename, "w") as file:
                file.write(header)

    def write(self, filecontents: str) -> None:
        # open file in append mode and write contents
        if MPI.COMM_WORLD.rank == 0:
            with open(self.filename, "a") as file:
                file.write(filecontents)


class MixedVTXWriter:
    """convenience class to ease output from mixed function spaces"""

    def __init__(
        self,
        comm: MPI.Comm,
        filebasenames: list,
        uη: Function,
        engine: str = "BPFile",
        mesh_policy: VTXMeshPolicy = VTXMeshPolicy.update,
    ) -> None:
        # save mixed function
        self.uη = uη

        # get number of components
        self.num_sub_spaces = uη.function_space.num_sub_spaces

        if self.num_sub_spaces == 2:
            # split mixed function into its components
            (self.u_out, self.η_out) = (
                self.uη.sub(0).collapse(),
                self.uη.sub(1).collapse(),
            )
            self.u_out.name = "u"
            self.η_out.name = "η"

            # create VTX writers
            self.vtx_u = VTXWriter(
                comm,
                filebasenames[0],
                [self.u_out],
                engine=engine,
                mesh_policy=mesh_policy,
            )
            self.vtx_η = VTXWriter(
                comm,
                filebasenames[1],
                [self.η_out],
                engine=engine,
                mesh_policy=mesh_policy,
            )

        elif self.num_sub_spaces == 3:
            # split mixed function into its components
            (self.u_out, self.p_out, self.q_out) = (
                self.uη.sub(0).collapse(),
                self.uη.sub(1).collapse(),
                self.uη.sub(2).collapse(),
            )
            self.u_out.name = "u"
            self.p_out.name = "ηtr"
            self.q_out.name = "ηdev"

            # create VTX writers
            self.vtx_u = VTXWriter(
                comm,
                filebasenames[0],
                [self.u_out],
                engine=engine,
                mesh_policy=mesh_policy,
            )
            self.vtx_η = VTXWriter(
                comm,
                filebasenames[1],
                [self.p_out, self.q_out],
                engine=engine,
                mesh_policy=mesh_policy,
            )

    def write(self, t: float) -> None:
        # split mixed function into its components
        if self.num_sub_spaces == 2:
            (u_out_i, η_out_i) = (
                self.uη.sub(0).collapse(),
                self.uη.sub(1).collapse(),
            )

            # overwrite arrays linked to VTX writers
            self.u_out.x.array[:] = u_out_i.x.array
            self.η_out.x.array[:] = η_out_i.x.array

        if self.num_sub_spaces == 3:
            (u_out_i, p_out_i, q_out_i) = (
                self.uη.sub(0).collapse(),
                self.uη.sub(1).collapse(),
                self.uη.sub(2).collapse(),
            )

            # overwrite arrays linked to VTX writers
            self.u_out.x.array[:] = u_out_i.x.array
            self.p_out.x.array[:] = p_out_i.x.array
            self.q_out.x.array[:] = q_out_i.x.array

        # write VTX files
        self.vtx_u.write(t)
        self.vtx_η.write(t)

    def close(self) -> None:
        # close VTX writers
        self.vtx_u.close()
        self.vtx_η.close()
