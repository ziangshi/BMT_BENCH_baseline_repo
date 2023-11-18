# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code that computes FVD for some empty frames.

The FVD for this setup should be around 131.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import frechet_video_distance as fvd
import torch
# Number of videos must be divisible by 16.
NUMBER_OF_VIDEOS = 2
VIDEO_LENGTH = 8


def main(argv):
  del argv
  with tf.Graph().as_default():


    ground_all_0 = torch.load('ablation_tensor/ground_0.pt')
    ground_all_1 = torch.load('ablation_tensor/ground_1.pt')

    fake_all_0 = torch.load('ablation_tensor/rvd2_10_0.pt')
    fake_all_1 = torch.load('ablation_tensor/rvd2_10_1.pt')

    stacked_tensor_ground = torch.vstack((ground_all_0.unsqueeze(0), ground_all_1.unsqueeze(0))).permute(0,2,3,4,1).numpy()
    stacked_tensor_fake = torch.vstack((fake_all_0.unsqueeze(0), fake_all_1.unsqueeze(0))).permute(0,2,3,4,1).numpy()

    #first_set_of_videos = tf.zeros([NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3])
    #second_set_of_videos = tf.ones([NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3]) * 255
    first_set_of_videos = tf.convert_to_tensor(stacked_tensor_ground)
    second_set_of_videos = tf.convert_to_tensor(stacked_tensor_fake)
    result = fvd.calculate_fvd(
        fvd.create_id3_embedding(fvd.preprocess(first_set_of_videos,
                                                (224, 224))),
        fvd.create_id3_embedding(fvd.preprocess(second_set_of_videos,
                                                (224, 224))))

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("FVD is: %.4f." % sess.run(result))


if __name__ == "__main__":
  tf.app.run(main)

#133.91 fvd follow the pre-defined parameter 16 video, 16 batch, 15 length seq
#inf rvd
#967.39 ivrnn
#1128.19 svg
#1663.79 rvd

#2——10 rvd 1533.47
#4_8 rvd   1437.33
#4_8 ivrnn 1934.38
#2_10 ivrnn