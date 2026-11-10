import os
import shutil
import bpy
from . import helper, blender_nerf_operator


# subset of frames operator class
class SubsetOfFrames(blender_nerf_operator.BlenderNeRF_Operator):
    '''Subset of Frames Operator'''
    bl_idname = 'object.subset_of_frames'
    bl_label = 'Subset of Frames SOF'

    def execute(self, context):
        scene = context.scene
        camera = scene.camera

        # check if camera is selected : next errors depend on an existing camera
        if camera == None:
            self.report({'ERROR'}, 'Be sure to have a selected camera!')
            return {'FINISHED'}

        # if there is an error, print first error message
        error_messages = self.asserts(scene, method='SOF')
        if len(error_messages) > 0:
           self.report({'ERROR'}, error_messages[0])
           return {'FINISHED'}

        output_data = self.get_camera_intrinsics(scene, camera)

        # clean directory name (unsupported characters replaced) and output path
        output_dir = bpy.path.clean_name(scene.sof_dataset_name)
        output_path = os.path.join(scene.save_path, output_dir)
        os.makedirs(output_path, exist_ok=True)

        if scene.logs: self.save_log_file(scene, output_path, method='SOF')
        if scene.splats: self.save_splats_ply(scene, output_path)

        # initial properties might have changed since set_init_props update
        scene.init_frame_step = scene.frame_step
        scene.init_frame_end = scene.frame_end
        scene.init_output_path = scene.render.filepath
        scene.init_active_camera = camera
        render_queue = []

        if scene.test_data:
            # testing transforms
            output_data['frames'] = self.get_camera_extrinsics(scene, camera, mode='TEST', method='SOF')
            self.save_json(output_path, 'transforms_test.json', output_data)
            if not (scene.splats and scene.splats_test_dummy):
                render_queue.append({
                    'camera': camera.name,
                    'directory': os.path.join(output_path, 'test'),
                    'frame_end': scene.frame_end,
                    'frame_step': scene.frame_step
                })

        if scene.val_data:
            # validation transforms
            output_val_data = self.get_camera_intrinsics(scene, camera)
            output_val_data['frames'] = self.get_camera_extrinsics(scene, camera, mode='VAL', method='SOF')
            self.save_json(output_path, 'transforms_val.json', output_val_data)
            render_queue.append({
                'camera': camera.name,
                'directory': os.path.join(output_path, 'val'),
                'frame_end': scene.frame_end,
                'frame_step': scene.frame_step
            })

        if scene.train_data:
            # training transforms
            output_data['frames'] = self.get_camera_extrinsics(scene, camera, mode='TRAIN', method='SOF')
            self.save_json(output_path, 'transforms_train.json', output_data)

            render_queue.append({
                'camera': camera.name,
                'directory': os.path.join(output_path, 'train'),
                'frame_end': scene.frame_end,
                'frame_step': scene.train_frame_steps
            })

        if render_queue:
            scene.rendering = (True, False, False)
            helper.set_render_background(scene)
            helper.start_render_queue(scene, render_queue)
        else:
            # compress dataset and remove folder (only keep zip)
            shutil.make_archive(output_path, 'zip', output_path) # output filename = output_path
            shutil.rmtree(output_path)

        return {'FINISHED'}