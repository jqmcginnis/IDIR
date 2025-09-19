from utils import general
from models import models
import os

data_dir = "/home/home/johannes_julian/NLST_red"
out_dir = "/home/home/johannes_julian/idir_out"

os.makedirs(out_dir,exist_ok=True)

case_id = 1
'''
(
    img_insp,
    img_exp,
    landmarks_insp,
    landmarks_exp,
    mask_exp,
    voxel_size,
) = general.load_image_DIRLab(case_id, "{}\Case".format(data_dir))
'''

(
    img_insp,
    img_exp,
    landmarks_insp,
    landmarks_exp,
    mask_exp,
    voxel_size,
) = general.load_NLST_data(case_id)

kwargs = {}
kwargs["verbose"] = True
kwargs["hyper_regularization"] = False
kwargs["jacobian_regularization"] = False
kwargs["bending_regularization"] = True
kwargs["network_type"] = "SIREN"  # Options are "MLP" and "SIREN"
kwargs["save_folder"] = out_dir + str(case_id)
kwargs["mask"] = mask_exp
# kwargs["epochs"] = 2 # changed ICLR

ImpReg = models.ImplicitRegistrator(img_exp, img_insp, **kwargs)
ImpReg.fit()
new_landmarks_orig, _ = general.compute_landmarks(
    ImpReg.network, landmarks_insp, image_size=img_insp.shape
)


_ = general.compute_deformation_field(ImpReg.network, image_size=img_insp.shape, voxel_size=voxel_size, output_path="deformation_field.nii.gz")

print(voxel_size)
accuracy_mean, accuracy_std = general.compute_landmark_accuracy(
    new_landmarks_orig, landmarks_exp, voxel_size=voxel_size
)

print("{} {} {}".format(case_id, accuracy_mean, accuracy_std))
