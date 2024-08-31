//
//  ContentView.swift
//  BasketballHighlights
//
//  Created by Swabhan Katkoori on 8/6/24.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        let columns = [GridItem(.adaptive(minimum: 75.00), spacing: 10)]
        
        Text("Customize Your Free Tier");
        
        ScrollView {
            LazyVGrid(columns: columns) {
                ForEach(0x1f600...0x1f679, id: \.self) { value in
                    GroupBox {
                        Text(emoji(value))
                            .font(.largeTitle)
                            .fixedSize()
                        Text(String(format: "%x", value))
                            .fixedSize()
                    }
                }
            }
            .padding()
            
        }
        
    }
}
    
private func emoji(_ value: Int) -> String {
      guard let scalar = UnicodeScalar(value) else { return "?" }
      return String(Character(scalar))
  }

#Preview {
    ContentView()
}
