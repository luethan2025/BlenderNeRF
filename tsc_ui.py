import bpy


# train, validation and test cameras ui class
class TSC_UI(bpy.types.Panel):
    '''Train, Validation and Test Cameras UI'''
    bl_idname = 'VIEW3D_PT_tsc_ui'
    bl_label = 'Train, Val, and Test Cameras TSC'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BlenderNeRF'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.alignment = 'CENTER'

        layout.use_property_split = True
        layout.prop(scene, 'ttc_nb_frames')
        layout.prop_search(scene, 'camera_train_target', scene, 'objects')
        layout.prop_search(scene, 'camera_val_target', scene, 'objects')
        layout.prop_search(scene, 'camera_test_target', scene, 'objects')

        layout.separator()
        layout.prop(scene, 'ttc_dataset_name')

        layout.separator()
        layout.operator('object.train_test_cameras', text='PLAY TSC')