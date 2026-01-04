import mesa

from .obs_agent import Human, ForcefulHuman, Line


class SimpleCanvas(mesa.visualization.VisualizationElement):
    local_includes = ["obs_server_model/simple_continuous_canvas.js"]
    portrayal_method = None
    canvas_height = 500
    canvas_width = 500

    def __init__(self, portrayal_method, canvas_height=500, canvas_width=500):
        self.portrayal_method = portrayal_method
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        new_element = "new Simple_Continous_Module({}, {})".format(
            self.canvas_width, self.canvas_height
        )
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        space_state = []
        for obj in model.schedule.agents:
            portrayal = self.portrayal_method(obj)
            if type(obj) is Line:
                x1, y1 = obj.start_pos
                x2, y2 = obj.end_pos
                x1 = (x1 - model.space.x_min) / \
                    (model.space.x_max - model.space.x_min)
                y1 = (y1 - model.space.y_min) / \
                    (model.space.y_max - model.space.y_min)
                x2 = (x2 - model.space.x_min) / \
                    (model.space.x_max - model.space.x_min)
                y2 = (y2 - model.space.y_min) / \
                    (model.space.y_max - model.space.y_min)
                portrayal["x1"] = x1
                portrayal["y1"] = y1
                portrayal["x2"] = x2
                portrayal["y2"] = y2
            if type(obj) is Human or type(obj) is ForcefulHuman:
                x, y = obj.pos
                x = (x - model.space.x_min) / \
                    (model.space.x_max - model.space.x_min)
                y = (y - model.space.y_min) / \
                    (model.space.y_max - model.space.y_min)
                portrayal["x"] = x
                portrayal["y"] = y
            space_state.append(portrayal)
        return space_state
