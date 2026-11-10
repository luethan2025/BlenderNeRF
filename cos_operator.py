import os
import shutil
import bpy
from . import helper, blender_nerf_operator


# global addon script variables
EMPTY_NAME = 'BlenderNeRF Sphere'
CAMERA_NAME = 'BlenderNeRF Camera'

# camera on sphere operator class
class CameraOnSphere(blender_nerf_operator.BlenderNeRF_Operator):
    '''Camera on Sphere Operator'''
    bl_idname = 'object.camera_on_sphere'
    bl_label = 'Camera on Sphere COS'

    def execute(self, context):
        scene = context.scene
        camera = scene.camera

        # check if camera is selected : next errors depend on an existing camera
        if camera == None:
            self.report({'ERROR'}, 'Be sure to have a selected camera!')
            return {'FINISHED'}

        # if there is an error, print first error message
        error_messages = self.asserts(scene, method='COS')
        if len(error_messages) > 0:
           self.report({'ERROR'}, error_messages[0])
           return {'FINISHED'}

        scale = scene.render.resolution_percentage / 100
        width = scene.render.resolution_x * scale
        height = scene.render.resolution_y * scale
        if width < 800 or height < 800:
            self.report({'ERROR'}, 'COS images must be at least 800x800 to apply center crop.')
            return {'FINISHED'}

        output_data = self.get_camera_intrinsics(scene, camera, crop_size=800)

        # clean directory name (unsupported characters replaced) and output path
        output_dir = bpy.path.clean_name(scene.cos_dataset_name)
        output_path = os.path.join(scene.save_path, output_dir)
        os.makedirs(output_path, exist_ok=True)

        if scene.logs: self.save_log_file(scene, output_path, method='COS')
        if scene.splats: self.save_splats_ply(scene, output_path)

        # initial property might have changed since set_init_props update
        scene.init_output_path = scene.render.filepath

        # other intial properties
        scene.init_sphere_exists = scene.show_sphere
        scene.init_camera_exists = scene.show_camera
        scene.init_frame_step = scene.frame_step
        scene.init_frame_end = scene.frame_end
        scene.init_active_camera = camera
        render_queue = []

        if scene.test_data:
            # testing transforms
            output_data['frames'] = self.get_camera_extrinsics(scene, camera, mode='TEST', method='COS')
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
            output_val_data = self.get_camera_intrinsics(scene, camera, crop_size=800)
            output_val_data['frames'] = self.get_camera_extrinsics(scene, camera, mode='VAL', method='COS')
            self.save_json(output_path, 'transforms_val.json', output_val_data)
            render_queue.append({
                'camera': camera.name,
                'directory': os.path.join(output_path, 'val'),
                'frame_end': scene.frame_end,
                'frame_step': scene.frame_step
            })

        if scene.train_data:
            if not scene.show_camera: scene.show_camera = True

            # train camera on sphere
            sphere_camera = scene.objects[CAMERA_NAME]
            sphere_output_data = self.get_camera_intrinsics(scene, sphere_camera, crop_size=800)
            scene.camera = sphere_camera

            # training transforms
            sphere_output_data['frames'] = self.get_camera_extrinsics(scene, sphere_camera, mode='TRAIN', method='COS')
            self.save_json(output_path, 'transforms_train.json', sphere_output_data)

            render_queue.append({
                'camera': sphere_camera.name,
                'directory': os.path.join(output_path, 'train'),
                'frame_end': scene.frame_start + scene.cos_nb_frames - 1,
                'frame_step': scene.frame_step
            })

        if render_queue:
            if not helper.set_center_crop(scene, crop_size=800):
                self.report({'ERROR'}, 'COS images must be rendered at least 800x800 to apply center crop.')
                return {'FINISHED'}
            helper.apply_cos_brightness(scene)
            scene.rendering = (False, False, True)
            helper.set_render_background(scene)
            helper.start_render_queue(scene, render_queue)
        else:
            # compress dataset and remove folder (only keep zip)
            shutil.make_archive(output_path, 'zip', output_path) # output filename = output_path
            shutil.rmtree(output_path)

        return {'FINISHED'}