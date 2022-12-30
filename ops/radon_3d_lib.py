import numpy as np
import matplotlib.pyplot as plt
import torch
import odl

# from odl.contrib.torch import OperatorFunction
from ops.ODLHelper import OperatorFunction

from .odl_lib import apply_angle_noise

class ParallelBeamGeometry3DOp(object):
  def __init__(self, img_size, num_angles, op_snr, angle_max=np.pi/3):
    self.img_size = img_size
    self.num_angles = num_angles
    self.angle_max = angle_max
    self.reco_space = odl.uniform_discr(
      min_pt=[-20, -20, -20],
      max_pt=[20, 20, 20],
      shape=[img_size, img_size, img_size],
      dtype='float32'
      )
      
    # Make a 3d single-axis parallel beam geometry with flat detector
    # Angles: uniformly spaced, n = 180, min = 0, max = pi
    # self.angle_partition = odl.uniform_partition(0, np.pi, 180)
    self.angle_partition = odl.uniform_partition(-angle_max, angle_max, num_angles)
    # Detector: uniformly sampled, n = (512, 512), min = (-30, -30), max = (30, 30)
    # self.detector_partition = odl.uniform_partition([-30, -30], [30, 30], [256, 256])
    # self.detector_partition = odl.tomo.parallel_beam_geometry(self.reco_space).det_partition
    self.detector_partition = odl.tomo.parallel_beam_geometry(self.reco_space,det_shape=(2*img_size,2*img_size)).det_partition
    self.geometry = odl.tomo.Parallel3dAxisGeometry(self.angle_partition, self.detector_partition)

    self.num_detectors_x, self.num_detectors_y = self.geometry.detector.shape

    self.angles = apply_angle_noise(self.geometry.angles, op_snr)
    self.optimizable_params = torch.tensor(self.angles, dtype=torch.float32)  # Convert to torch.Tensor.  
    
    self.op = odl.tomo.RayTransform(self.reco_space, self.geometry, impl='astra_cuda')

    self.fbp = odl.tomo.analytic.filtered_back_projection.fbp_op(self.op)

  def __call__(self, x):
    return OperatorFunction.apply(self.op, x)

  def pinv(self, y):
    return OperatorFunction.apply(self.fbp, y)

class ParallelBeamGeometry3DOpBroken(ParallelBeamGeometry3DOp):
  def __init__(self, clean_operator, op_snr):
    super().__init__(clean_operator.img_size, clean_operator.num_angles, op_snr, clean_operator.angle_max)

    self.optimizable_params = torch.tensor(clean_operator.geometry.angles, dtype=torch.float32)

    self.angles = apply_angle_noise(clean_operator.geometry.angles, op_snr)
    # angle partition is changed to not be uniform
    self.angle_partition = odl.discr.nonuniform_partition(np.sort(self.angles))

    self.geometry = odl.tomo.Parallel3dAxisGeometry(self.angle_partition, self.detector_partition)

    self.num_detectors_x, self.num_detectors_y = self.geometry.detector.shape

    self.op = odl.tomo.RayTransform(self.reco_space, self.geometry, impl='astra_cuda')
    self.fbp = odl.tomo.analytic.filtered_back_projection.fbp_op(self.op)


class ParallelBeamGeometry3DOpAngles(ParallelBeamGeometry3DOp):
  def __init__(self, img_size, angles, op_snr, angle_max=np.pi/3):
    super().__init__(img_size, angles.shape[0], op_snr,angle_max)

    self.optimizable_params = torch.tensor(self.geometry.angles, dtype=torch.float32)

    self.angles = angles
    # angle partition is changed to not be uniform
    self.angle_partition = odl.discr.nonuniform_partition(np.sort(self.angles))

    # TODO: change following axes to get rotation around another axis
    self.geometry = odl.tomo.Parallel3dAxisGeometry(self.angle_partition, self.detector_partition,
    axis=(1,0,0),det_axes_init=[(1, 0, 0), (0, 1, 0)],det_pos_init=(0,0,1))


    self.num_detectors_x, self.num_detectors_y = self.geometry.detector.shape

    self.op = odl.tomo.RayTransform(self.reco_space, self.geometry, impl='astra_cuda')
    self.fbp = odl.tomo.analytic.filtered_back_projection.fbp_op(self.op)


def unit_test():
  img_size = 64
  num_angles = 60
  A = ParallelBeamGeometry3DOp(img_size, num_angles, np.inf)

  x = torch.rand([img_size, img_size, img_size])
  y = A(x)
  x_hat = A.pinv(y)
  print (x.shape)
  print (y.shape)
  print(x_hat.shape)

if __name__ == "__main__":
  unit_test()