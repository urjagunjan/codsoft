import java.util.Scanner;

public class Chatbot {

    public static String getResponse(String userInput) {
        String lowerCaseInput = userInput.toLowerCase();

        if (lowerCaseInput.contains("hello") || lowerCaseInput.contains("hi")) {
            return "Hello there! How can I help you today?";
        } else if (lowerCaseInput.contains("how are you")) {
            return "I am just a bot, but I'm doing great! Thanks for asking.";
        } else if (lowerCaseInput.contains("what is your name") || lowerCaseInput.contains("who are you")) {
            return "I am a simple rule-based chatbot  CRISTIE created to assist you.";
        } else if (lowerCaseInput.contains("bye") || lowerCaseInput.contains("goodbye")) {
            return "Goodbye! Have a great day.";
        } else if (lowerCaseInput.contains("help")) {
            return "You can ask me things like 'hello', 'how are you', 'what is your name', or say 'bye' to exit.";
        } else {
            return "I'm sorry, I don't understand that. Please try asking something else or type 'help'.";
        }
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Chatbot: Hello! I'm a simple chatbot. Type 'bye' to exit.");

        while (true) {
            System.out.print("You: ");
            String userInput = scanner.nextLine();

            if (userInput.toLowerCase().contains("bye") || userInput.toLowerCase().contains("goodbye")) {
                System.out.println("Chatbot: Goodbye! Have a great day.");
                break;
            }

            String response = getResponse(userInput);
            System.out.println("Chatbot: " + response);
        }

        scanner.close();
    }
}

