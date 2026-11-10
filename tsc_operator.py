import os
import shutil
import bpy
from . import helper, blender_nerf_operator


# three-split cameras operator class
class TrainValTestCameras(blender_nerf_operator.BlenderNeRF_Operator):
    '''Three-Split Cameras Operator'''
    bl_idname = 'object.train_test_cameras'
    bl_label = 'Three-Split Cameras TSC'

    def execute(self, context):
        scene = context.scene
        train_camera = scene.camera_train_target
        val_camera = scene.camera_val_target
        test_camera = scene.camera_test_target

        # if there is an error, print first error message
        error_messages = self.asserts(scene, method='TTC')
        if len(error_messages) > 0:
           self.report({'ERROR'}, error_messages[0])
           return {'FINISHED'}

        # clean directory name (unsupported characters replaced) and output path
        output_dir = bpy.path.clean_name(scene.ttc_dataset_name)
        output_path = os.path.join(scene.save_path, output_dir)
        os.makedirs(output_path, exist_ok=True)

        output_train_data = self.get_camera_intrinsics(scene, train_camera) if scene.train_data else None
        output_val_data = self.get_camera_intrinsics(scene, val_camera) if scene.val_data else None
        output_test_data = self.get_camera_intrinsics(scene, test_camera) if scene.test_data else None

        if scene.logs: self.save_log_file(scene, output_path, method='TTC')
        if scene.splats: self.save_splats_ply(scene, output_path)

        # initial properties might have changed since set_init_props update
        scene.init_output_path = scene.render.filepath
        scene.init_frame_step = scene.frame_step
        scene.init_frame_end = scene.frame_end
        scene.init_active_camera = scene.camera
        render_queue = []

        if scene.test_data:
            # testing transforms
            output_test_data['frames'] = self.get_camera_extrinsics(scene, test_camera, mode='TEST', method='TTC')
            self.save_json(output_path, 'transforms_test.json', output_test_data)
            if not (scene.splats and scene.splats_test_dummy):
                render_queue.append({
                    'camera': test_camera.name,
                    'directory': os.path.join(output_path, 'test'),
                    'frame_end': scene.frame_end,
                    'frame_step': scene.frame_step
                })

        if scene.val_data:
            # validation transforms
            output_val_data['frames'] = self.get_camera_extrinsics(scene, val_camera, mode='VAL', method='TTC')
            self.save_json(output_path, 'transforms_val.json', output_val_data)
            render_queue.append({
                'camera': val_camera.name,
                'directory': os.path.join(output_path, 'val'),
                'frame_end': scene.frame_end,
                'frame_step': scene.frame_step
            })

        if scene.train_data:
            # training transforms
            output_train_data['frames'] = self.get_camera_extrinsics(scene, train_camera, mode='TRAIN', method='TTC')
            self.save_json(output_path, 'transforms_train.json', output_train_data)

            render_queue.append({
                'camera': train_camera.name,
                'directory': os.path.join(output_path, 'train'),
                'frame_end': scene.frame_start + scene.ttc_nb_frames - 1,
                'frame_step': scene.frame_step
            })

        if render_queue:
            scene.rendering = (False, True, False)
            helper.set_render_background(scene)
            helper.start_render_queue(scene, render_queue)
        else:
            # compress dataset and remove folder (only keep zip)
            shutil.make_archive(output_path, 'zip', output_path) # output filename = output_path
            shutil.rmtree(output_path)

        return {'FINISHED'}