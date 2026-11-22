import unittest
from computer_observer import validate_result, ALLOWED_OPERATIONS


class ObserverValidationTests(unittest.TestCase):
    def test_only_four_read_only_operations_are_allowed(self):
        self.assertEqual(
            ALLOWED_OPERATIONS,
            {
                "workstation", "blender_status",
                "blender_scene_snapshot", "blender_project_snapshot",
            },
        )

    def test_valid_workstation(self):
        data = {
            "hostname":"DESKTOP",
            "os":"Windows",
            "architecture":"AMD64",
            "processor":"CPU",
            "python":"3.13.0"
        }
        self.assertEqual(validate_result("workstation", data), data)

    def test_rejects_extra_workstation_fields(self):
        data = {
            "hostname":"DESKTOP", "os":"Windows",
            "architecture":"AMD64", "processor":"CPU",
            "python":"3.13.0", "extra":"unexpected"
        }
        with self.assertRaises(ValueError):
            validate_result("workstation", data)

    def test_valid_blender_status(self):
        data = {
            "installed":True, "executable":"D:\\\\blender\\\\blender.exe",
            "version":"Blender 5.3.0", "version_error":None,
            "running":False
        }
        self.assertEqual(validate_result("blender_status", data), data)

    def test_rejects_non_boolean_blender_flags(self):
        data = {
            "installed":"true", "executable":"D:\\\\blender\\\\blender.exe",
            "version":"Blender", "version_error":None, "running":False
        }
        with self.assertRaises(ValueError):
            validate_result("blender_status", data)

    def test_rejects_unknown_operation(self):
        with self.assertRaises(ValueError):
            validate_result("execute_command", {})

    def test_rejects_wrong_result_type(self):
        with self.assertRaises(ValueError):
            validate_result("workstation", [])


if __name__ == "__main__":
    unittest.main()
